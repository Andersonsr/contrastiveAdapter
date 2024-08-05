import numpy as np
import torch
import torch.nn as nn
import clip
from projectionMLP import BaseAdapter, ProjectionMLP
device = "cuda" if torch.cuda.is_available() else "cpu"


class DualAdapter(BaseAdapter):
    def __init__(self, in_dim, out_dim, text_embeddings, initial_logit_scale):
        super(DualAdapter, self).__init__()
        self.imageAdapter = ProjectionMLP(in_dim, out_dim, 1)
        self.imageAdapter.to(device)
        self.textAdapter = ProjectionMLP(in_dim, out_dim, 1)
        self.textAdapter.to(device)
        self.text_embeddings = text_embeddings
        self.logit_scale = nn.Parameter(torch.ones([]) * initial_logit_scale)

    def forward(self, batch):
        images_embeddings = batch[0].to(device, torch.float32)
        labels = batch[1].to(device)

        images_embeddings = self.imageAdapter(images_embeddings)
        images_embeddings = images_embeddings / images_embeddings.norm(dim=1, keepdim=True)
        text_embeddings_resized = self.textAdapter(self.text_embeddings)
        text_embeddings_resized = text_embeddings_resized / text_embeddings_resized.norm(dim=1, keepdim=True)
        logits = (images_embeddings @ text_embeddings_resized.T) * self.logit_scale.exp()
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def predict(self, batch):
        images_embeddings = batch[0].to(device, torch.float32)
        images_embeddings = self.imageAdapter(images_embeddings)
        images_embeddings = images_embeddings / images_embeddings.norm(dim=1, keepdim=True)
        text_embeddings_resized = self.textAdapter(self.text_embeddings)
        text_embeddings_resized = text_embeddings_resized / text_embeddings_resized.norm(dim=1, keepdim=True)
        logits = (images_embeddings @ text_embeddings_resized.T) * self.logit_scale.exp()
        return logits.softmax(dim=1).argmax(dim=1)

