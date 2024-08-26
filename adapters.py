import random

import torch
import torch.nn as nn
from projection_heads import TwoLayerHead,  ResidualHead, TwoLayerHeadNB, TwoLayerHeadDO, ResidualLearnableHead
import foundation_models
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"


class DualAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, text_embeddings, initial_logit_scale, variant='default'):
        super(DualAdapter, self).__init__()
        if variant == 'nb':
            self.imageAdapter = TwoLayerHeadNB(in_dim, out_dim)
            self.textAdapter = TwoLayerHeadNB(in_dim, out_dim)

        elif variant == 'do':
            self.imageAdapter = TwoLayerHeadDO(in_dim, out_dim)
            self.textAdapter = TwoLayerHeadDO(in_dim, out_dim)

        else:
            self.imageAdapter = TwoLayerHead(in_dim, out_dim)
            self.textAdapter = TwoLayerHead(in_dim, out_dim)

        self.imageAdapter.to(device)
        self.textAdapter.to(device)
        self.text_embeddings = text_embeddings
        self.logit_scale = nn.Parameter(torch.ones([]) * initial_logit_scale)

    def forward(self, batch):
        images_embeddings = batch[0].to(device, torch.float32)

        images_embeddings = self.imageAdapter(images_embeddings)
        images_embeddings = images_embeddings / images_embeddings.norm(dim=1, keepdim=True)
        text_embeddings_resized = self.textAdapter(self.text_embeddings)
        text_embeddings_resized = text_embeddings_resized / text_embeddings_resized.norm(dim=1, keepdim=True)
        return (images_embeddings @ text_embeddings_resized.T) * self.logit_scale.exp()

    def predict(self, batch):
        self.eval()
        logits = self.forward(batch)
        return logits.softmax(dim=1).argmax(dim=1)


class SingleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, text_embeddings, initial_logit_scale, variant='default'):
        super(SingleAdapter, self).__init__()
        if variant == 'nb':
            self.imageAdapter = TwoLayerHeadNB(in_dim, out_dim)
        elif variant == 'do':
            self.imageAdapter = TwoLayerHeadDO(in_dim, out_dim)
        else:
            self.imageAdapter = TwoLayerHead(in_dim, out_dim)

        self.imageAdapter.to(device)
        self.text_embeddings = text_embeddings
        self.logit_scale = nn.Parameter(torch.ones([]) * initial_logit_scale)

    def forward(self, batch):
        images_embeddings = batch[0].to(device, torch.float32)
        images_embeddings = self.imageAdapter(images_embeddings)
        images_embeddings = images_embeddings / images_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = self.text_embeddings
        text_embeddings_resized = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        return (images_embeddings @ text_embeddings_resized.T) * self.logit_scale.exp()

    def predict(self, batch):
        self.eval()
        logits = self.forward(batch)
        return logits.softmax(dim=1).argmax(dim=1)


class CustomAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, text_embeddings, initial_logit_scale):
        super(CustomAdapter, self).__init__()
        self.imageAdapter = ResidualHead(in_dim, out_dim, alpha)
        self.imageAdapter.to(device)
        self.text_embeddings = text_embeddings
        self.logit_scale = nn.Parameter(torch.ones([]) * initial_logit_scale)

    def forward(self, batch):
        images_embeddings = batch[0].to(device, torch.float32)
        images_embeddings = self.imageAdapter(images_embeddings)
        images_embeddings = images_embeddings / images_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=1, keepdim=True)
        return (images_embeddings @ text_embeddings.T) * self.logit_scale.exp()

    def predict(self, batch):
        self.eval()
        logits = self.forward(batch)
        return logits.softmax(dim=1).argmax(dim=1)


class ContrastiveResidualAdapter(nn.Module):
    def __init__(self, in_dim):
        super(ContrastiveResidualAdapter, self).__init__()
        self.foundation = foundation_models.CLIP(device)
        self.foundation.load_model()
        self.imageAdapter = ResidualLearnableHead(in_dim)
        self.textAdapter = ResidualLearnableHead(in_dim)
        self.logit_scale = nn.Parameter(self.foundation.backbone.logit_scale)

    def forward(self, batch):
        image_features = batch[0].to(device, torch.float32).squeeze()
        text_features = batch[1].to(device, torch.float32)
        c = random.randint(0, text_features.shape[1]-1)
        text_features = text_features[:, c, :]
        # print(text_features.shape, image_features.shape, c)

        # resized features logits
        image_features = self.imageAdapter.forward(image_features)
        text_features = self.textAdapter.forward(text_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return (image_features @ text_features.T) * (self.logit_scale.exp())

    def image_projection(self, embeddings):
        self.eval()
        return self.imageAdapter(embeddings)

    def text_projection(self, embeddings):
        self.eval()
        return self.textAdapter(embeddings)

    def train_epoch(self, train_loader, optim):
        self.train()
        epoch_losses = []

        for batch in train_loader:
            optim.zero_grad()
            logits = self.forward(batch)
            targets = torch.arange(len(batch[0])).to(device)
            # print(logits.shape, targets.shape)
            i_loss = nn.CrossEntropyLoss()(logits, targets)
            t_loss = nn.CrossEntropyLoss()(logits.T, targets)
            loss = i_loss + t_loss
            loss.backward()
            optim.step()
            epoch_losses.append(loss.detach().cpu())
        return np.mean(epoch_losses)

    def val_epoch(self, val_loader):
        self.eval()
        epoch_losses = []

        for batch in val_loader:
            logits = self.forward(batch)
            targets = torch.arange(len(batch[0])).to(device)
            i_loss = nn.CrossEntropyLoss()(logits, targets)
            t_loss = nn.CrossEntropyLoss()(logits.T, targets)
            loss = i_loss + t_loss
            epoch_losses.append(loss.detach().cpu())
        return np.mean(epoch_losses)


class ContrastiveAdapter(nn.Module):
    def __init__(self, input_size, output_size):
        super(ContrastiveAdapter, self).__init__()
        self.foundation = foundation_models.CLIP(device)
        self.foundation.load_model()
        self.imageAdapter = ResidualHead(input_size, output_size, alpha=0.6, )
        self.textAdapter = ResidualHead(input_size, output_size, alpha=0.6, )
        self.logit_scale = nn.Parameter(self.foundation.backbone.logit_scale)

    def forward(self, batch):
        self.train()
        image_features = batch[0].to(device, torch.float32)
        text_features = batch[1].to(device, torch.float32)

        # resized features logits
        image_features = self.imageAdapter.forward(image_features)
        text_features = self.textAdapter.forward(text_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return (image_features @ text_features.T) * (self.logit_scale.exp())

    def predict(self, batch):
        self.eval()
        image_features = batch[0].to(device, torch.float32)
        text_features = batch[1].to(device, torch.float32)

        # resized features logits
        image_features = self.imageAdapter.forward(image_features)
        text_features = self.textAdapter.forward(text_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits = (image_features @ text_features.T)
        return logits

    def image_projection(self, embeddings):
        self.eval()
        return self.imageAdapter(embeddings)

    def text_projection(self, embeddings):
        self.eval()
        return self.textAdapter(embeddings)
