import torch
import torchvision.datasets.fgvc_aircraft
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import models
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def extract_features_geo(model, dataset='dataset/geoVQA.xlsx'):
    output = {'image_path': [], 'text': [], 'image_features': [], 'text_features': [], 'text_length': [], 'width': [],
              'height': []}
    df = pd.read_excel(dataset)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        path = row['image']
        image = Image.open(path)
        width, height = image.size
        caption = row['gt_text']
        with torch.no_grad():
            img_features = model.visual_embedding(row['image'])
            text_features = model.language_embedding(caption)

        output['image_features'].append(img_features.cpu())
        output['text_features'].append(text_features.cpu())
        output['text_length'].append(len(caption))
        output['width'].append(width)
        output['height'].append(height)
        output['image_path'].append(row['image'])
        output['text'].append(caption)

    with open('dataset/embeddings/clip_embeddings.pkl', 'wb') as f:
        pickle.dump(output, f)


def extract_features_torchvision(model, data, save_path):
    output = {'image_features': [], 'label': [],}
    for image, label in tqdm(data):
        output['label'].append(label)
        image = model.vision_preprocess(image).unsqueeze(0).to(device)
        image_feature = model.backbone.encode_image(image)
        features = image_feature.detach().cpu()
        output['image_features'].append(features)

    df = pd.DataFrame(output)
    with open(save_path, 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    model = models.CLIP(device)
    model.load_model()
    data = torchvision.datasets.stanford_cars.StanfordCars(
        root='datasets-torchvis/stanford_cars',
        split='test',
        )

    extract_features_torchvision(model, data, 'datasets-torchvis/embeddings/cars_test.pkl')





