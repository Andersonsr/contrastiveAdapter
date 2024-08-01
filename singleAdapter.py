import numpy as np
import torch
import torch.nn as nn
import clip
from projectionMLP import BaseAdapter, ProjectionMLP
device = torch.device("cuda" if torch.cuda.is_available() else "")


def identity(x):
    return x


class SingleAdapter(BaseAdapter):
    def __init__(self, in_dim, out_dim, alpha, clip_encoder, texts):
        super(SingleAdapter, self).__init__()
        self.imageAdapter = ProjectionMLP(in_dim, out_dim, alpha)
        self.imageAdapter.to(device)
        # self.textAdapter = ProjectionMLP(in_dim, out_dim, alpha)
        # self.textAdapter.to(device)
        self.texts = texts
        self.clip_backbone, _ = clip.load(clip_encoder)
        self.logit_scale = self.clip_backbone.logit_scale

    def encode_texts(self):
        texts = torch.cat([clip.tokenize(text) for text in self.texts]).to(device)
        with torch.no_grad():
            texts = self.clip_backbone.encode_text(texts).to(torch.float32)
        # texts = self.textAdapter(texts)
        return texts

    def forward(self, batch):
        images_embeddings = batch[0].to(device, torch.float32)
        labels = batch[1].to(device)

        images_embeddings = self.imageAdapter(images_embeddings)
        text_embeddings = self.encode_texts()
        images_embeddings = images_embeddings / images_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        logits = (images_embeddings @ text_embeddings.T) * self.logit_scale.exp()

        loss = nn.CrossEntropyLoss()(logits, labels)
        pred = logits.softmax(dim=1).argmax(dim=1)
        accuracy = sum((pred == labels)*1) / labels.shape[0]
        return loss, accuracy



