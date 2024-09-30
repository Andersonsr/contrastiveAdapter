import random
import torch
import torch.nn as nn
from projection_heads import TwoLayerHead,  ResidualHead, TwoLayerHeadNB, TwoLayerHeadDO, ResidualLearnableHead, ResidualDynamicHead
from logitNormLoss import LogitNormLoss
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
from embeddingsLoader import CaptionDataset


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
    def __init__(self, in_dim, initial_residual_ratio, initial_logit_scale, trainable_residual_ratio=True, logit_norm=False):
        super(ContrastiveResidualAdapter, self).__init__()
        self.imageAdapter = ResidualLearnableHead(in_dim, initial_residual_ratio, trainable_residual_ratio)
        self.textAdapter = ResidualLearnableHead(in_dim, initial_residual_ratio, trainable_residual_ratio)
        self.logit_scale = nn.Parameter(initial_logit_scale)
        self.logit_norm = logit_norm

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
        return self.imageAdapter(embeddings.to(device, torch.float32))

    def text_projection(self, embeddings):
        self.eval()
        return self.textAdapter(embeddings.to(device, torch.float32))

    def loss(self, logits, targets):
        if self.logit_norm:
            i_loss = LogitNormLoss(device, )(logits, targets)
            t_loss = LogitNormLoss(device, )(logits, targets)
            return i_loss + t_loss

        else:
            i_loss = nn.CrossEntropyLoss()(logits, targets)
            t_loss = nn.CrossEntropyLoss()(logits.T, targets)
            return i_loss + t_loss

    def train_epoch(self, train_loader, optim):
        self.train()
        epoch_losses = []
        for batch in train_loader:
            optim.zero_grad()
            logits = self.forward(batch)
            targets = torch.arange(len(batch[0])).to(device)
            loss = self.loss(logits, targets)
            loss.backward()
            optim.step()
            self.textAdapter.residual = nn.Parameter(torch.clamp(self.textAdapter.residual, min=0, max=1))
            self.imageAdapter.residual = nn.Parameter(torch.clamp(self.imageAdapter.residual, min=0, max=1))
            epoch_losses.append(loss.detach().cpu())
        return np.mean(epoch_losses)

    def val_epoch(self, val_loader):
        self.eval()
        epoch_losses = []

        for batch in val_loader:
            logits = self.forward(batch)
            targets = torch.arange(len(batch[0])).to(device)
            loss = self.loss(logits, targets)
            epoch_losses.append(loss.detach().cpu())
        return np.mean(epoch_losses)


class DynamicContrastiveResidualAdapter(ContrastiveResidualAdapter):
    def __init__(self, in_dim, initial_residual_ratio, logit_scale, trainable_residual_ratio=True, bottleneck_factor=4,
                 depth=0):
        super(ContrastiveResidualAdapter, self).__init__()
        self.imageAdapter = ResidualDynamicHead(in_dim, initial_residual_ratio, trainable_residual_ratio,
                                                bottleneck_factor, depth)
        self.textAdapter = ResidualDynamicHead(in_dim, initial_residual_ratio, trainable_residual_ratio,
                                               bottleneck_factor, depth)
        self.logit_scale = nn.Parameter(logit_scale)


class SigAdapter(nn.Module):
    def __init__(self, in_dim, initial_residual_ratio, initial_bias, initial_logit_scale, multi_positive=False,
                 use_logit_bias=True, reduction=4):
        super(SigAdapter, self).__init__()
        self.imageAdapter = ResidualLearnableHead(in_dim, initial_residual_ratio, trainable_residual_ratio=False,
                                                  bottleneck_reduction_ratio=reduction)
        self.textAdapter = ResidualLearnableHead(in_dim, initial_residual_ratio, trainable_residual_ratio=False,
                                                 bottleneck_reduction_ratio=reduction)
        self.logit_scale = nn.Parameter(initial_logit_scale)
        self.logit_bias = nn.Parameter(initial_bias)
        self.multi_positive = multi_positive
        self.use_logit_bias = use_logit_bias
        # print(self.logit_bias, self.logit_scale)

    def forward(self, batch):
        image_features = batch[0].to(device, torch.float32).squeeze()
        text_features = batch[1].to(device, torch.float32)
        c = random.randint(0, text_features.shape[1]-1)
        if not self.multi_positive:
            text_features = text_features[:, c, :]
        else:
            text_features = torch.reshape(text_features, (text_features.shape[0] * text_features.shape[1], text_features.shape[2]))

        image_features = self.imageAdapter.forward(image_features)
        text_features = self.textAdapter.forward(text_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return ((image_features @ text_features.T) * (self.logit_scale.exp()) +
                self.logit_bias) if self.use_logit_bias else 0

    def image_projection(self, embeddings):
        self.eval()
        return self.imageAdapter(embeddings.to(device, torch.float32))

    def text_projection(self, embeddings):
        self.eval()
        return self.textAdapter(embeddings.to(device, torch.float32))

    def targets(self, n, positives=5):
        if self.multi_positive:
            targets = torch.ones(n, n*positives) * -1
            for i in range(n):
                targets[i, i*positives:i*positives+5] += 2
            return targets
        else:
            return (2 * torch.eye(n)) - torch.ones(n)

    def train_epoch(self, train_loader, optim):
        self.train()
        epoch_losses = []
        for batch in train_loader:
            n = len(batch[0])
            optim.zero_grad()
            logits = self.forward(batch)
            targets = self.targets(n).to(device)
            loss = -torch.sum(nn.LogSigmoid()(logits * targets)) / n
            loss.backward()
            optim.step()
            epoch_losses.append(loss.detach().cpu())

        return np.mean(epoch_losses)

    def val_epoch(self, val_loader):
        self.eval()
        epoch_losses = []

        for batch in val_loader:
            n = len(batch[0])
            logits = self.forward(batch)
            targets = self.targets(n).to(device)
            loss = -torch.sum(nn.LogSigmoid()(logits * targets)) / n
            loss.backward()
            epoch_losses.append(loss.detach().cpu())

        return np.mean(epoch_losses)


if __name__ == '__main__':
    adapter = SigAdapter(768, 0.3, torch.ones([])*-10, torch.ones([])*4, True, ).to(device)
    dataset = 'coco_openCLIP'
    val_dataset = CaptionDataset(f'datasets_torchvision/embeddings/{dataset}_val.pkl')
    val_loader, val_indices = val_dataset.get_loader(shuffle=False, batch_size=20)
    for batch in val_loader:
        adapter.forward(batch)
        break


