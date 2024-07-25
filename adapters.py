import os.path
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from geoVQAdataset import GeoVQADataset
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchsummary import summary
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "")


class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, alpha):
        super(Adapter, self).__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(in_dim, in_dim // 4)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_dim // 4, out_dim)
        self.relu2 = nn.ReLU()

    def forward(self, embeddings):
        x = self.fc1(embeddings)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


class ContrastiveAdapter(nn.Module):
    def __init__(self, input_size, output_size, alpha):
        super(ContrastiveAdapter, self).__init__()
        self.imageAdapter = Adapter(input_size, output_size, alpha)
        self.textAdapter = Adapter(input_size, output_size, alpha)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, batch):
        image_features = batch[0]
        text_features = batch[1]

        # resized features logits
        image_features = self.imageAdapter.forward(image_features)
        text_features = self.textAdapter.forward(text_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits = (text_features @ image_features.T) * (self.logit_scale.exp())

        # contrastive loss
        targets = torch.arange(len(batch[0])).to(device)
        image_loss = nn.CrossEntropyLoss()(logits.T, targets)
        text_loss = nn.CrossEntropyLoss()(logits, targets)
        contrastive_loss = (image_loss + text_loss) / 2

        return contrastive_loss

    def train_epoch(self, train_loader, optim):
        self.train()

        epoch_losses = []
        for batch in train_loader:
            optim.zero_grad()
            batch_loss = self.forward(batch)
            batch_loss.backward()
            optim.step()
            epoch_losses.append(batch_loss.detach().item())

        return np.sum(epoch_losses)/len(epoch_losses)

    def val_epoch(self, val_loader):
        self.eval()
        epoch_losses = []
        for batch in val_loader:
            batch_loss = self.forward(batch)
            epoch_losses.append(batch_loss.detach().item())

        return np.sum(epoch_losses)/len(epoch_losses)

    def image_projection(self, embeddings):
        self.eval()
        return self.imageAdapter(embeddings)

    def text_projection(self, embeddings):
        self.eval()
        return self.textAdapter(embeddings)


def train(in_dim, out_dim, inputs_path, batch_size, epochs, output_name):
    dataset = GeoVQADataset([inputs_path], device)
    model = ContrastiveAdapter(in_dim, out_dim, 0.0)
    model.to(device)
    train_loader, val_loader = dataset.get_loaders(batch_size=batch_size)

    training_loss = []
    validation_loss = []

    summary(model.imageAdapter, input_size=(1, 768))
    optim = Adam(model.parameters(), lr=1e-5)

    for i in tqdm(range(1, epochs+1)):
        epoch_loss = model.train_epoch(train_loader, optim)
        training_loss.append(epoch_loss)
        validation_loss.append(model.val_epoch(val_loader))

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': training_loss[-1],
        },
        f'checkpoints/{output_name}_{out_dim}_{batch_size}.pt')

    loss_log = {'val_loss': validation_loss, 'training_loss': training_loss}
    df = pd.DataFrame.from_dict(loss_log)
    df.to_csv(f'loss_logs/{output_name}_{out_dim}_{batch_size}.csv', index=False)

    plt.plot(training_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    min_loss = np.argmin(validation_loss)
    plt.text(min_loss, training_loss[min_loss], f'{training_loss[min_loss]:.3}')
    plt.text(min_loss, validation_loss[min_loss], f'{validation_loss[min_loss]:.3}')

    plt.title(f'Loss embedding size {out_dim}, batch size: {batch_size} ')
    plt.legend()
    plt.savefig(f'plots/{output_name}_{out_dim}_{batch_size}.png')
    plt.clf()


def resize_features(checkpoint_path, in_dim, out_dim, input_embeddings):
    assert os.path.exists(path), f'No checkpoint found at {path}'
    model = ContrastiveAdapter(in_dim, out_dim, 0)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    dataset = GeoVQADataset(input_embeddings, device)
    _, test_loader = dataset.get_loaders(train_ratio=0, batch_size=len(dataset), shuffle=False)

    for batch in test_loader:
        images = model.image_projection(batch[0]).cpu()
        texts = model.text_projection(batch[1]).cpu()
        results = {'image_features': images, 'text_features': texts}
        with open(f'GeoVQA-dataset/resized_embeddings/long-clip_dropout_{out_dim}_{batch_size}.pkl', 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    in_dim = 768
    epochs = 200
    path = 'GeoVQA-dataset/embeddings/embeddings_longclip.pkl'
    name = 'dropout-fc1'

    for batch_size in [400]:
        for out_dim in [256, 128, 64, 32, 16]:
            checkpoint_path = f'checkpoints/dropout-fc1_{out_dim}_{batch_size}.pt'
            resize_features(checkpoint_path, in_dim, out_dim, path)



