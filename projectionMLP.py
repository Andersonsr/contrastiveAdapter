import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "")


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, alpha):
        super(ProjectionMLP, self).__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(in_dim, in_dim // 4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_dim // 4, out_dim)
        self.relu2 = nn.ReLU()

    def forward(self, embeddings):
        x = self.fc1(embeddings)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        if embeddings.shape[1] == x.shape[1]:
            # residual
            x = self.alpha * x + (1 - self.alpha) * embeddings
        return x


class BaseAdapter(nn.Module):
    def __init__(self):
        super(BaseAdapter, self).__init__()

    def train_epoch(self, train_loader, optim):
        self.train()
        epoch_losses = []
        epoch_accuracies = []

        for batch in train_loader:
            optim.zero_grad()
            batch_loss, batch_acc = self.forward(batch)
            batch_loss.backward()
            optim.step()
            epoch_losses.append(batch_loss.detach().cpu())
            if batch_acc is not None:
                epoch_accuracies.append(batch_acc.detach().cpu())
        acc = np.mean(epoch_accuracies) if len(epoch_accuracies) > 0 else None
        return np.mean(epoch_losses), acc

    def val_epoch(self, val_loader):
        self.eval()
        epoch_losses = []
        epoch_accuracies = []

        for batch in val_loader:
            batch_loss, batch_acc = self.forward(batch)
            epoch_losses.append(batch_loss.detach().item())
            if batch_acc is not None:
                epoch_accuracies.append(batch_acc.detach().cpu())

        acc = np.mean(epoch_accuracies) if len(epoch_accuracies) > 0 else None
        return np.mean(epoch_losses), acc
