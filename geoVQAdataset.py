import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class GeoVQADataset(Dataset):
    def __init__(self, file_path, device, text_column='max_len 50'):
        self.images = []
        self.texts = []
        self.device = device

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for i in range(len(data['image_features'])):
            img = data['image_features'][i]
            txt = data['text_features'][i]
            if len(data['text_features'][i].shape) == 3:
                # blip variant
                self.texts.append(txt[:, 0, :])
                similarity = (img @ txt[:, 0, :].t())
                self.images.append(img[0, torch.argmax(similarity, dim=1).item(), :])

            elif len(data['image_features'][i].shape) == 2:
                # clip variant
                self.images.append(img[0, :])
                self.texts.append(txt[0, :])

    def __getitem__(self, index):
        return self.images[index].to(self.device, torch.float32),  self.texts[index].to(self.device, torch.float32)

    def __len__(self):
        return len(self.images)

    def get_loaders(self, random_seed=59, batch_size=64, train_ratio=0.7, shuffle=True):
        size = len(self)
        indices = list(range(size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        split = int(size * train_ratio)
        train_sampler = torch.utils.data.SubsetRandomSampler(indices[:split])
        test_sampler = torch.utils.data.SubsetRandomSampler(indices[split:])
        train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=test_sampler)
        return train_loader, test_loader


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else '')
    dataset = GeoVQADataset(['dataset/embeddings/embeddings_longclip.pkl',
                            'dataset/embeddings/summarize_long_embeddings.pkl'], device)
    train_loader, test_loader = dataset.get_loaders()
    print(len(dataset))
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Texts batch shape: {train_labels.size()}")


