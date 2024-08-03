import time
import matplotlib.pyplot as plt
from util import get_labels, CUSTOM_TEMPLATES, encode_texts, plot_curves
import torch
import clip
from tqdm import tqdm
from embeddingsLoader import EmbeddingDataset
from torch.optim import Adam
import pandas as pd
from singleAdapter import SingleAdapter
from early_stopper import EarlyStopper
from sklearn.metrics import accuracy_score
device = torch.device("cuda" if torch.cuda.is_available() else "")


def run_training():
    logit_scale = clip.load('ViT-L/14')[0].logit_scale
    for dataset_alias in ['flowers', 'aircraft', 'cars']:
        template = CUSTOM_TEMPLATES[dataset_alias]
        categories = get_labels(dataset_alias)
        texts = [template.format(c) for c in categories]
        texts = encode_texts(texts)

        for k in [1, 2, 4, 8, 16]:
            dataset = EmbeddingDataset(f'datasets-torchvis/embeddings/{dataset_alias}_train.pkl', k=k)
            train_loader = dataset.get_loaders(train_ratio=1.0, batch_size=32)[0]
            dataset_test = EmbeddingDataset(f'datasets-torchvis/embeddings/{dataset_alias}_train.pkl')
            test_loader = dataset_test.get_loaders(train_ratio=0.0)[1]
            model = SingleAdapter(768, 768, 0.6, texts, logit_scale)
            optim = Adam(model.parameters(), lr=0.00001)
            stopper = EarlyStopper(patience=200, minimal_improvement=0.01, objective='minimize', save_option='last')
            identifier = f'vision only k={k} d=768 {dataset_alias}'
            training_losses = []
            validation_losses = []

            print(f'training {identifier}')
            time.sleep(1)

            for i in tqdm(range(1, 200)):
                training_loss = model.train_epoch(train_loader, optim)
                validation_loss = model.val_epoch(test_loader)

                training_losses.append(training_loss)
                validation_losses.append(validation_loss)

                model_dict = {'epoch': i,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optim.state_dict(),
                              'loss': training_loss
                              }

                if not stopper.continue_training(training_loss, model_dict):
                    break

            checkpoint = stopper.model_to_save()
            assert checkpoint is not None, 'Checkpoint is None'
            torch.save(checkpoint, f'checkpoints/{identifier}.pt')
            loss_log = {'val_loss': validation_losses, 'training_loss': training_losses}
            df = pd.DataFrame.from_dict(loss_log)
            df.to_csv(f'loss_logs/{identifier}', index=False)
            plot_curves(training_losses, validation_losses, identifier, 'loss')


def run_comparison():
    result = {'cars': [], 'aircraft': [], 'flowers': []}
    logit_scale = clip.load('ViT-L/14')[0].logit_scale
    for dataset_alias in ['flowers', 'aircraft', 'cars']:
        template = CUSTOM_TEMPLATES[dataset_alias]
        categories = get_labels(dataset_alias)
        texts = [template.format(c) for c in categories]
        texts = encode_texts(texts)
        model = SingleAdapter(768, 768, 0.6, texts, logit_scale)
        dataset = EmbeddingDataset(f'datasets-torchvis/embeddings/{dataset_alias}_test.pkl')
        test_loader = dataset.get_loaders(train_ratio=0.0, batch_size=len(dataset))[1]

        for k in [1, 2, 4, 8, 16]:
            path = f'checkpoints/classification/vision only k={k} d=768 {dataset_alias}.pt'
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            for batch in test_loader:
                predicts = model.predict(batch).detach().cpu()
                gt_labels = batch[1].detach().cpu()
                acc = accuracy_score(gt_labels, predicts)
                result[dataset_alias].append(acc)

    pd.DataFrame.from_dict(result).to_csv('comparison.csv', index=False)
    ks = [1, 2, 4, 8, 16]
    print(result)
    zeroshot = {'cars': 77.3, 'aircraft': 36.1, 'flowers': 78.7, }
    for alias in ['cars', 'flowers', 'aircraft']:
        plt.plot(ks, result[alias], '-o', label='custom adapter')
        plt.plot(0, zeroshot[alias]/100,'x', label='CLIP zero-shot')
        plt.xlabel('k-shots')
        plt.ylabel('accuracy')
        plt.title(f'few shot accuracy custom CLIP-Adapter {alias} dataset')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    run_comparison()


