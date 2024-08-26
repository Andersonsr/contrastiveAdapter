import clip
import numpy as np
import torch
from embeddingsLoader import CaptionDataset
from tqdm import tqdm
import clip
device = torch.device("cuda" if torch.cuda.is_available() else "")


def evaluate_image_text(path, n=5):
    model, _ = clip.load('ViT-L/14')
    t = model.logit_scale
    dataset = CaptionDataset(path, n_captions=n)
    loader, indices = dataset.get_loader(batch_size=5000, shuffle=False)
    result = []
    for batch in loader:
        images = batch[0].to(device).squeeze()
        captions = batch[1].to(device).flatten(start_dim=0, end_dim=1)
        images = images / images.norm(dim=-1, keepdim=True)
        captions = captions / captions.norm(dim=-1, keepdim=True)
        sim = (images @ captions.T) * t.exp()
        print(sim.shape)
        rank = sim.argsort(descending=True, dim=1) // 5
        for i in range(images.shape[0]):
            matches = (rank[i, :] == i)
            result.append(matches.nonzero().squeeze()[0])

    result = torch.FloatTensor(result)
    r1 = (result < 1).nonzero().shape[0] / result.shape[0]
    r5 = (result < 5).nonzero().shape[0] / result.shape[0]
    r10 = (result < 10).nonzero().shape[0] / result.shape[0]
    print(r1, r5, r10)


if __name__ == '__main__':
    evaluate_image_text('datasets_torchvision/embeddings/coco_ViTL_val.pkl')

