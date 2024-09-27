import open_clip
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from abc import ABC, abstractmethod
import clip
from long_clip.model import longclip


class Model(ABC):
    def __init__(self, device):
        self.backbone = None
        self.vision_preprocess = None
        self.language_preprocess = None
        self.device = device

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def visual_embedding(self, image_path):
        pass

    @abstractmethod
    def language_embedding(self, text):
        pass

    def similarity(self, text_features, image_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).max()


class Blip2(Model):
    def load_model(self):
        self.backbone, self.vision_preprocess, self.language_preprocess = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=self.device)

    def visual_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.vision_preprocess["eval"](image).unsqueeze(0).to(self.device)
        return self.backbone.extract_features({'image': image}, mode='image').image_embeds

    def language_embedding(self, text):
        return self.backbone.extract_features({'text_input': [text]}, mode='text').text_embeds

    def similarity(self, text_features, image_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features[:, 0, :].T).max()


class CLIP(Model):
    def load_model(self, encoder='ViT-L/14'):
        self.backbone, self.vision_preprocess = clip.load(encoder, device=self.device)
        self.language_preprocess = clip.tokenize

    def visual_embedding(self, image_path):
        with torch.no_grad():
            image = self.vision_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
            return self.backbone.encode_image(image)

    def language_embedding(self, text):
        with torch.no_grad():
            text = clip.tokenize(text, context_length=77, truncate=True).to(self.device)
            return self.backbone.encode_text(text)


class OpenCoCa(Model):
    def visual_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.vision_preprocess(image).unsqueeze(0).to(self.device)
        return self.backbone.encode_image(image)

    def language_embedding(self, text):
        text = self.language_preprocess(text)
        # print(text.shape, text)
        text = text[:, :76].to(self.device)

        return self.backbone.encode_text(text)

    def load_model(self):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="mscoco_finetuned_laion2B-s13B-b90k",
            device=self.device
        )
        self.language_preprocess = open_clip.get_tokenizer('ViT-L-14')


class OpenCLIP(Model):
    def visual_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.vision_preprocess(image).unsqueeze(0).to(self.device)
        return self.backbone.encode_image(image)

    def language_embedding(self, text):
        text = self.language_preprocess(text)
        return self.backbone.encode_text(text.to(self.device))

    def load_model(self):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            device=self.device
        )
        self.language_preprocess = open_clip.get_tokenizer('ViT-L-14')


class LongCLIP(Model):
    def load_model(self):
        self.backbone, self.vision_preprocess = longclip.load(
            "./long_clip/checkpoints/longclip-L.pt", device=self.device)

    def visual_embedding(self, image_path):
        image = self.vision_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.backbone.encode_image(image)

    def language_embedding(self, text):
        with torch.no_grad():
            return self.backbone.encode_text(longclip.tokenize(text).to(self.device))



