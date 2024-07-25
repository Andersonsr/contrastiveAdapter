import torch
from PIL import Image
import pandas as pd
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
import pickle
import models

if __name__ == '__main__':
    # load model
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model = models.LongCLIP(device)
    model.load_model()
    output = {'image_path': [], 'text': [],  'image_features': [], 'text_features': [], 'text_length': [], 'width': [], 'height': []}
    df = pd.read_excel('dataset/geoVQA.xlsx')
    summarized = pd.read_excel('dataset/summarized_text.xlsx')

    for i, row in tqdm(df.iterrows(), total=len(df)):
        image = Image.open(row['image'])
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

    # output_dict = {}
    # for column in summarized.columns:
    #     output_dict[column] = []
    #
    # for i, row in tqdm(summarized.iterrows(), total=len(summarized['max_len 50'])):
    #     for column in list(summarized.columns):
    #         with torch.no_grad():
    #             if isinstance(row[column], str):
    #                 text_feature = model.language_embedding(row[column])
    #                 output_dict[column].append(text_feature)
    #             else:
    #                 output_dict[column].append('')

    with open('GeoVQA-dataset/embeddings/summarize_long_embeddings.pkl', 'wb') as f:
        pickle.dump(output, f)
