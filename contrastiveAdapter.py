import torch
import torch.nn as nn
from projectionMLP import BaseAdapter, ProjectionMLP
import numpy as np
import os
import pickle
from geoVQAdataset import GeoVQADataset
device = torch.device("cuda" if torch.cuda.is_available() else "")


class ContrastiveAdapter(BaseAdapter):
    def __init__(self, input_size, output_size):
        super(ContrastiveAdapter, self).__init__()
        self.imageAdapter = ProjectionMLP(input_size, output_size, 1)
        self.textAdapter = ProjectionMLP(input_size, output_size, 1)
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

        return contrastive_loss, None

    def predict(self, batch):
        pass

    def image_projection(self, embeddings):
        self.eval()
        return self.imageAdapter(embeddings)

    def text_projection(self, embeddings):
        self.eval()
        return self.textAdapter(embeddings)


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








