import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "")


class ResidualHead(nn.Module):
    def __init__(self, in_dim, out_dim, alpha):
        super(ResidualHead, self).__init__()
        self.alpha = alpha
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim // 4, out_dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, embeddings):
        x = self.model(embeddings)
        if embeddings.shape[1] == x.shape[1]:
            x = self.alpha * x + (1 - self.alpha) * embeddings
        return x


class ResidualLearnableHead(nn.Module):
    def __init__(self, in_dim):
        super(ResidualLearnableHead, self).__init__()
        self.residual = nn.Parameter(torch.ones([]) * 0.6)
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim),
            nn.ReLU()
        )

    def forward(self, embeddings):
        x = self.model(embeddings)
        x = self.residual * x + embeddings
        return x


class TwoLayerHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TwoLayerHead, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, out_dim),
            nn.ReLU()
        )

    def forward(self, embeddings):
        return self.model(embeddings)


class TwoLayerHeadNB(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TwoLayerHeadNB, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim // 4, out_dim, bias=False),
            nn.ReLU()
        )

    def forward(self, embeddings):
        return self.model(embeddings)


class TwoLayerHeadDO(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TwoLayerHeadDO, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, out_dim),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, embeddings):
        return self.model(embeddings)