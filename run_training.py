import argparse
import torch
from tqdm import tqdm
from embeddingsLoader import EmbeddingDataset
from geoVQAdataset import GeoVQADataset
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
from cars_categories import names
import pandas as pd
from contrastiveAdapter import ContrastiveAdapter
from singleAdapter import SingleAdapter
from early_stopper import EarlyStopper
device = torch.device("cuda" if torch.cuda.is_available() else "")


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'flowers': 'a photo of a {}, a type of flower.',
    'aircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'cars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}


def get_labels(dataset_name):
    if dataset_name == 'aircraft':
        with open('datasets-torchvis/fgvc_aircraft/fgvc-aircraft-2013b/data/variants.txt') as f:
            lines = f.readlines()
            return [line.strip() for line in lines]

    if dataset_name == 'flowers':
        data = pd.read_csv('datasets-torchvis/flowers102/flowers-102/oxford_flower_102_name.csv')
        return data['name'].tolist()

    if dataset_name == 'cars':
        return names


def plot_curves(training, validation, output_name, type):
    plt.plot(training, label=f'training {type}')
    plt.plot(validation, label=f'validation {type}')

    if type == 'loss':
        min_loss = np.argmin(validation)
        plt.text(min_loss, training[min_loss], f'{training[min_loss]:.3}')
        plt.text(min_loss, validation[min_loss], f'{validation[min_loss]:.3}')

    elif type == 'accuracy':
        max_acc = np.argmax(validation)
        plt.text(max_acc, training[max_acc], f'{training[max_acc]:.3}')
        plt.text(max_acc, validation[max_acc], f'{validation[max_acc]:.3}')

    plt.title(f'{type} curves {output_name}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default='10', help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-i', '--input_dim', type=int, default=768, help='input embedding dimension')
    parser.add_argument('-o', '--output_dim', type=int, default=768, help='output embedding dimension')
    parser.add_argument('-k', '--k_shots', type=int, default=0, help='number of shots')
    parser.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='[0,1] learned features proportion, when out_dim = in_dim')
    parser.add_argument('--data_path', type=str, default='datasets-torchvis/embeddings/aircraft_train.pkl',
                        help='path to embedding dataset')
    parser.add_argument('--dataset', type=str, default='aircraft', choices=['aircraft', 'flowers', 'cars'],
                        help='dataset alias when using supervised learning')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/experiment.pt',
                        help='path to save checkpoint')
    parser.add_argument('--loss_log', type=str, default='loss_logs/test.csv', help='path to save losses')
    parser.add_argument('--learning_objective', type=str, default='supervised',
                        choices=['contrastive', 'supervised'], help='type of learning objective')
    parser.add_argument('--identifier', type=str, default='experiment', help='experiment identifier')
    args = parser.parse_args()

    assert os.path.exists(args.data_path), 'data path does not exist'

    if args.learning_objective == 'supervised':
        template = CUSTOM_TEMPLATES[args.dataset]
        categories = get_labels(args.dataset)
        texts = [template.format(c) for c in categories]
        model = SingleAdapter(args.input_dim, args.output_dim, args.alpha, 'ViT-L/14', texts)

        if args.k_shots > 0:
            dataset = EmbeddingDataset(args.data_path, k=args.k_shots)
            train_loader = dataset.get_loaders(train_ratio=1)[0]

        else:
            dataset = EmbeddingDataset(args.data_path)
            train_loader = dataset.get_loaders()[0]

        dataset = EmbeddingDataset(args.data_path.replace('train', 'test'))
        val_loader = dataset.get_loaders()[1]

    else:
        model = ContrastiveAdapter(args.input_dim, args.output_dim)
        dataset = GeoVQADataset(args.data_path)
        train_loader, val_loader = dataset.get_loaders()[:2]

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    optim = Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=5, minimal_improvement=0.01, objective='minimize', save_option='last')

    for i in tqdm(range(1, args.epochs)):
        training_loss, training_acc = model.train_epoch(train_loader, optim)
        validation_loss, validation_acc = model.val_epoch(val_loader)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        if validation_acc is not None:
            training_accuracies.append(training_acc)
            validation_accuracies.append(validation_acc)

        model_dict = {'epoch': args.epochs, 'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optim.state_dict(),
                      'loss': training_losses[-1]}

        if not stopper.continue_training(training_loss, model_dict):
            torch.save(stopper.model_to_save(), args.checkpoint_path)
            break

    loss_log = {'val_loss': validation_losses, 'training_loss': training_losses}
    df = pd.DataFrame.from_dict(loss_log)
    df.to_csv(args.loss_log, index=False)

    plot_curves(training_losses, validation_losses, args.identifier, 'loss')
    plot_curves(training_accuracies, validation_accuracies, args.identifier, 'accuracy')


