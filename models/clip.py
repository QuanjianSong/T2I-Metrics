import torch
import clip
from PIL import Image


class CLIPModel:
    def __init__(self, model_path="ViT-B/32", device="cuda"):
        self.model, self.preprocess = clip.load(model_path, device)
        self.model.eval()
        self.device = device

    def __call__(self, input):
        with torch.no_grad():
            # breakpoint()
            if isinstance(input, Image.Image):
                    image_input = self.preprocess(input).unsqueeze(0).to(self.device)
                    feat = self.model.encode_image(image_input)
            if isinstance(input, str):
                    text = clip.tokenize([input]).to(self.device)
                    feat = self.model.encode_text(text)
        return feat