import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from cars_categories import names
import pandas as pd

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
        return data['Name'].tolist()

    if dataset_name == 'cars':
        return names


def encode_texts(texts):
    texts = torch.cat([clip.tokenize(text) for text in texts]).to(device)
    clip_backbone, _ = clip.load("ViT-L/14")
    with torch.no_grad():
        texts = clip_backbone.encode_text(texts).to(torch.float32)
    return texts


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
    plt.savefig(f'{output_name}.png')
    plt.clf()
