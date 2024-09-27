import open_clip
import torchvision.datasets as dset
import torch
from PIL import Image
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    data = dset.CocoCaptions(root=f'datasets_torchvision/coco_2017/val'
                                  f''
                                  f'2017',
                             annFile=f'datasets_torchvision/coco_2017/annotations/captions_val2017.json', )

    fine_tuned_model, _, fine_preprocess = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k",
        device=device
    )

    zero_shot_model, _, zero_preprocess = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="laion2B-s13B-b90k",
        device=device
    )

    for i, row in enumerate(data):
        img = row[0]

        with torch.no_grad(), torch.cuda.amp.autocast():
            zero_caption = zero_shot_model.generate(zero_preprocess(img).unsqueeze(0).to(device))
            fine_caption = fine_tuned_model.generate(fine_preprocess(img).unsqueeze(0).to(device))
            print(row[1])
            print(open_clip.decode(zero_caption[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
            print(open_clip.decode(fine_caption[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))
            print()
        if i > 9:
            break

