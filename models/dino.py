
import torch
from transformers import AutoImageProcessor, AutoModel


class DINOv2Model:
    def __init__(self, model_path="facebook/dinov2-large", device="cuda"):
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    def __call__(self, img_list):
        with torch.no_grad():
            inputs = self.processor(images=img_list, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            img_feats = outputs.last_hidden_state
            img_feats = img_feats.mean(dim=1)
        return img_feats