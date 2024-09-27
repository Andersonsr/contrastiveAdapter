import clip
import numpy as np
import torch
from embeddingsLoader import CaptionDataset
from tqdm import tqdm
from adapters import ContrastiveResidualAdapter, SigAdapter, DynamicContrastiveResidualAdapter
import matplotlib.pyplot as plt
import clip
import open_clip
device = torch.device("cuda" if torch.cuda.is_available() else "")


def evaluate_image_text(path, temperature, n=5, mode='one'):
    dataset = CaptionDataset(path, n_captions=n)
    loader, indices = dataset.get_loader(batch_size=5000, shuffle=False)
    result = []
    for batch in loader:
        images = batch[0].to(device).squeeze()
        captions = batch[1].to(device).flatten(start_dim=0, end_dim=1)
        images = images / images.norm(dim=-1, keepdim=True)
        captions = captions / captions.norm(dim=-1, keepdim=True)
        sim = (images @ captions.T) * temperature.exp()
        # print(sim.shape)
        rank = sim.argsort(descending=True, dim=1) // 5
        for i in range(images.shape[0]):
            matches = (rank[i, :] == i)
            result.append(matches.nonzero().squeeze())

    results = torch.stack(result, dim=0)
    if mode == 'one':
        # check only the first caption, 1 correct caption
        results = results.squeeze()
        r1 = (results[:, 0] < 1).nonzero().shape[0] / results.shape[0]
        r5 = (results[:, 0] < 5).nonzero().shape[0] / results.shape[0]
        r10 = (results[:, 0] < 10).nonzero().shape[0] / results.shape[0]
        return {'R@1': r1, 'R@5': r5, 'R@10': r10}

    elif mode == 'all':
        # check all captions, 5 correct captions
        r5 = (results < 5).nonzero().shape[0] / (results.shape[0] * results.shape[1])
        r10 = (results < 10).nonzero().shape[0] / (results.shape[0] * results.shape[1])
        r15 = (results < 15).nonzero().shape[0] / (results.shape[0] * results.shape[1])
        return {'R@5': r5, 'R@10': r10, 'R@15': r15}


if __name__ == '__main__':
    model, _ = clip.load('ViT-L/14')
    results = {}
    mode = 'one'
    # result = evaluate_image_text('datasets_torchvision/embeddings/coco_ViTL_val.pkl',
    #                              model.logit_scale, mode=mode)
    # results['CLIP zero shot'] = result.values()
    paths = ['coco_openCLIP_val',
             'OpenCLIP_residual_adapter_0.3',
             'OpenCLIP_sig_adapter_0.4_nobias',
             'OpenCLIP_sig_adapter_0.3',]

    names = ['openCLIP',
             'Residual Adapter alpha=0.3',
             'Sig Adapter alpha=0.4 no bias',
             'Sig adapter alpha=0.3',]

    for i, name in enumerate(paths):
        if 'adapter_' in name:
            if 'sig' in name:
                adapter = SigAdapter(768, 0.2, torch.ones([]), torch.ones([]), ).to(device)

            else:
                adapter = ContrastiveResidualAdapter(768, 0.2, model.logit_scale, False)

            checkpoint = torch.load(f'checkpoints/contrastive/{name}.pt')
            adapter.load_state_dict(checkpoint['model_state_dict'])
            result = evaluate_image_text(f'datasets_torchvision/embeddings/{name}.pkl',
                                         adapter.logit_scale, mode=mode)
        else:
            result = evaluate_image_text(f'datasets_torchvision/embeddings/{name}.pkl',
                                         model.logit_scale, mode=mode)

        results[names[i]] = result.values()

    print(results)
    k_ints = []
    for k in result.keys():
        k_ints.append(int(k.split('@')[1]))

    for k, v in results.items():
        plt.plot(k_ints, v, '-o', label=k)

    plt.legend()
    plt.xlabel('k')
    plt.ylabel('R@k')
    plt.title('COCO Retrieval with Open CLIP encoders, 5/25000')
    plt.show()

