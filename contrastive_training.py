import os
import pickle
import time
import torch
from adapters import ContrastiveAdapter, ContrastiveResidualAdapter
from tqdm import tqdm
from torch.optim import Adam
from util import plot_curves
from geoVQAdataset import GeoVQADataset
from embeddingsLoader import CaptionDataset
from early_stopping import EarlyStopping
device = torch.device("cuda" if torch.cuda.is_available() else "")


def resize_features(checkpoint_path: str, in_dim: int, out_dim: int, input_embeddings: str, batch_size: int):
    assert os.path.exists(checkpoint_path), f'No checkpoint found at {checkpoint_path}'
    model = ContrastiveAdapter(in_dim, out_dim)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    dataset = GeoVQADataset(input_embeddings)
    test_loader = dataset.get_loaders(train_ratio=0, batch_size=len(dataset), shuffle=False)[1]

    for batch in test_loader:
        images = model.image_projection(batch[0]).cpu()
        texts = model.text_projection(batch[1]).cpu()
        results = {'image_features': images, 'text_features': texts}
        with open(f'dataset/resized_embeddings/long-clip_dropout_{out_dim}_{batch_size}.pkl', 'wb') as f:
            pickle.dump(results, f)


def run_training():
    train_dataset = CaptionDataset('datasets_torchvision/embeddings/coco_ViTL_train.pkl')
    val_dataset = CaptionDataset('datasets_torchvision/embeddings/coco_ViTL_val.pkl')
    train_loader, train_indices = train_dataset.get_loader(shuffle=False)
    val_loader, val_indices = val_dataset.get_loader(shuffle=False)
    es = EarlyStopping(patience=10, minimal_improvement=0.01, objective='minimize', save_option='last')
    identifier = f'clip_residual'
    model = ContrastiveResidualAdapter(768).to(device)
    optim = Adam(model.parameters(), lr=0.00001)
    training_losses = []
    validation_losses = []

    print(f'training {identifier}')
    time.sleep(1)

    for i in tqdm(range(1, 3)):
        training_loss = model.train_epoch(train_loader, optim)
        validation_loss = model.val_epoch(val_loader)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        model_dict = {'epoch': 200,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': training_losses[-1]
                      }
        es.update(validation_loss, model_dict)
        if es.stop:
            break

    torch.save(es.model_to_save(), f'checkpoints/contrastive/{identifier}.pt')
    plot_curves(training_losses, validation_losses, identifier, 'loss')


if __name__ == '__main__':
    run_training()
