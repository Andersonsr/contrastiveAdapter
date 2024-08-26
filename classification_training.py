import time
import matplotlib.pyplot as plt
from util import get_labels, CUSTOM_TEMPLATES, encode_texts, plot_curves
import torch
import clip
import numpy as np
from tqdm import tqdm
from embeddingsLoader import EmbeddingDataset
from torch.optim import Adam
import pandas as pd
from torch import nn
from adapters import DualAdapter, CustomAdapter, SingleAdapter
from early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "")


def train_epoch(model, train_loader, optim):
    model.train()
    epoch_losses = []

    for batch in train_loader:
        optim.zero_grad()
        logits = model.forward(batch)
        batch_loss = nn.CrossEntropyLoss()(logits, batch[1].to(device))
        batch_loss.backward()
        optim.step()
        epoch_losses.append(batch_loss.detach().cpu())
    return np.mean(epoch_losses)


def val_epoch(model, val_loader):
    model.eval()
    epoch_losses = []

    for batch in val_loader:
        logits = model.forward(batch)
        batch_loss = nn.CrossEntropyLoss()(logits, batch[1].to(device))
        epoch_losses.append(batch_loss.detach().cpu())

    return np.mean(epoch_losses)


def run_training(type):
    assert type in ['custom', 'single', 'dual', 'single_nb', 'dual_nb', 'single_do', 'dual_do'], \
        'type must be in [custom, single, dual, single_nb, dual_nb, single_do, dual_do]'
    logit_scale = clip.load('ViT-L/14')[0].logit_scale
    for dataset_alias in ['flowers', 'cars', 'aircraft']:
        template = CUSTOM_TEMPLATES[dataset_alias]
        categories = get_labels(dataset_alias)
        texts = [template.format(c) for c in categories]
        texts = encode_texts(texts, model='ViT-L/14').to(device)

        # for k in [1, 2, 4, 8, 16]:
        dataset = EmbeddingDataset(f'datasets_torchvision/embeddings/{dataset_alias}_ViTL_train.pkl')
        train_loader = dataset.get_loaders(train_ratio=1.0, batch_size=32)[0]
        dataset_test = EmbeddingDataset(f'datasets_torchvision/embeddings/{dataset_alias}_ViTL_test.pkl')
        test_loader = dataset_test.get_loaders(train_ratio=0.0)[1]
        for out_dim in [768]:
            variant = type.split('_')[-1]
            if type == 'single':
                model = SingleAdapter(768, out_dim, texts, logit_scale, variant=variant)
                stopper = EarlyStopping(patience=10, minimal_improvement=0.01, objective='minimize',
                                        save_option='last')
            elif type == 'custom':
                model = CustomAdapter(768, out_dim, 0.6, texts, logit_scale)
                stopper = EarlyStopping(patience=10, minimal_improvement=0.01, objective='minimize',
                                        save_option='last')
            else:
                model = DualAdapter(768, out_dim, texts, logit_scale, variant=variant)
                stopper = EarlyStopping(patience=10, minimal_improvement=0.01, objective='minimize', save_option='last')

            optim = Adam(model.parameters(), lr=0.00001)
            identifier = f'{type} adapter d={out_dim} {dataset_alias} ViT-L14'
            training_losses = []
            validation_losses = []

            print(f'training {identifier}')
            time.sleep(1)

            for i in tqdm(range(1, 200)):
                training_loss = train_epoch(model, train_loader, optim)
                validation_loss = val_epoch(model, test_loader)

                training_losses.append(training_loss)
                validation_losses.append(validation_loss)

                model_dict = {'epoch': i,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optim.state_dict(),
                              'loss': training_loss
                              }

                stopper.update(validation_loss, model_dict)
                if stopper.stop:
                    break

            checkpoint = stopper.model_to_save()
            assert checkpoint is not None, 'Checkpoint is None'
            torch.save(checkpoint, f'checkpoints/classification/{identifier}.pt')
            loss_log = {'val_loss': validation_losses, 'training_loss': training_losses}
            df = pd.DataFrame.from_dict(loss_log)
            df.to_csv(f'loss_logs/{identifier}', index=False)
            plot_curves(training_losses, validation_losses, identifier, 'loss')


def run_comparison():
    logit_scale = clip.load('RN50')[0].logit_scale

    # bars
    fig, ax = plt.subplots(layout='constrained')
    results = {'custom': [], 'single': [], 'dual': [], }

    for dataset_alias in ['flowers', 'cars', 'aircraft']:
        template = CUSTOM_TEMPLATES[dataset_alias]
        categories = get_labels(dataset_alias)
        texts = [template.format(c) for c in categories]
        texts = encode_texts(texts, model='ViT-L/14')
        dataset = EmbeddingDataset(f'datasets_torchvision/embeddings/{dataset_alias}_ViTL_test.pkl')
        test_loader = dataset.get_loaders(train_ratio=0.0, batch_size=len(dataset))[1]
        # results = {'custom': [], 'single': [], 'dual': [], }
        ks = [0]
        ds = [768]
        types = ['custom', 'single', 'dual', ]
        for d in ds:
            for k in ks:
                for t in types:
                    path = f'checkpoints/classification/{t} adapter d={d} {dataset_alias} ViT-L14.pt'
                    checkpoint = torch.load(path)
                    variant = t.split('_')[-1]
                    if 'custom' in t:
                        model = CustomAdapter(768, d, 0.6, texts, logit_scale)
                    elif 'single' in t:
                        model = SingleAdapter(768, d, texts, logit_scale, variant=variant)
                    else:
                        model = DualAdapter(768, d, texts, logit_scale, variant=variant)
                    # print('loading model', path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    for batch in test_loader:
                        # unico batch
                        predicts = model.predict(batch).detach().cpu()
                        gt_labels = batch[1].detach().cpu()
                        acc = accuracy_score(gt_labels, predicts)
                        results[str(t)].append(acc)
        # for key in results.keys():
        #     plt.plot(ks, results[key], '-o', label=key)
        #
        # plt.xlabel('k-shots')
        # plt.ylabel('accuracy')
        # plt.title(f'k-shot accuracy {dataset_alias} dataset ViT-L14')
        # plt.legend()
        # plt.show()

    x = np.arange(3)  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    for k, v in results.items():
        offset = width * multiplier
        ax.bar(x + offset, v, width, label=k, )
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('accuracy')
    ax.set_xlabel('dataset')
    ax.set_title('models accuracy by dataset')
    ax.set_xticks(x + width, ['flowers', 'cars', 'aircraft'])
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0.0, 1.1)
    plt.show()


if __name__ == '__main__':
    # run_training('dual_do')
    # run_training('single_do')
    # run_training('custom')
    run_comparison()
    # per_class_acc()

