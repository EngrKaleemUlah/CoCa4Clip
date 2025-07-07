# from transformers import AutoProcessor, AutoModel
# import torch


# class CoCaWrapper(torch.nn.Module):
#     def __init__(self, model_name="CoCa-ViT-B-32", pretrained="laion2B-s13B-b90k"):
#         super().__init__()
#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)

#     def forward_image(self, image_tensor):
#         # image_tensor: BxCxHxW
#         inputs = self.processor(images=image_tensor, return_tensors="pt").to(
#             image_tensor.device
#         )
#         outputs = self.model(**inputs)
#         return outputs.image_embeds  # BxD

#     def forward_text(self, text_list):
#         inputs = self.processor(
#             text=text_list, return_tensors="pt", padding=True, truncation=True
#         ).to(self.model.device)
#         outputs = self.model(**inputs)
#         return outputs.text_embeds  # BxD

# import torch
# import torch.nn as nn
# import open_clip


# class CoCaWrapper(nn.Module):
#     def __init__(self, model_name="hf-hub:laion/CoCa-ViT-L-14-laion2B-s13B-b90k"):
#         super().__init__()
#         self.model, _, self.preprocess = open_clip.create_model_and_transforms(
#             model_name
#         )
#         self.tokenizer = open_clip.get_tokenizer(model_name)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def forward_image(self, image_tensor):
#         # image_tensor: [B, C, H, W]
#         image = self.preprocess(image_tensor).to(self.device)
#         with torch.no_grad():
#             image_emb = self.model.encode_image(image)
#         return image_emb  # [B, D]

#     def forward_text(self, text_list):
#         tokens = self.tokenizer(text_list).to(self.device)
#         with torch.no_grad():
#             text_emb = self.model.encode_text(tokens)
#         return text_emb  # [B, D]

# import torch
# import torch.nn as nn
# import open_clip


# class CoCaWrapper(nn.Module):
#     def __init__(self, model_name="coca_ViT-L-14", pretrained="laion2B-s13B-b90k"):
#         super().__init__()
#         self.model, _, self.preprocess = open_clip.create_model_and_transforms(
#             model_name=model_name, pretrained=pretrained
#         )
#         self.tokenizer = open_clip.get_tokenizer(model_name)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def forward_image(self, image_tensor):
#         image = self.preprocess(image_tensor).to(self.device)
#         with torch.no_grad():
#             image_emb = self.model.encode_image(image)
#         return image_emb

#     def forward_text(self, text_list):
#         tokens = self.tokenizer(text_list).to(self.device)
#         with torch.no_grad():
#             text_emb = self.model.encode_text(tokens)
#         return text_emb
import torch
import torch.nn as nn
import open_clip
from torchvision.transforms.functional import to_pil_image


class CoCaWrapper(nn.Module):
    def __init__(self, model_name="coca_ViT-L-14", pretrained="laion2B-s13B-b90k"):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.model.logit_scale.item())

    def forward_image(self, image_tensor, video_frame=None):
        """
        image_tensor: [B, C, H, W] in torch.Tensor format
        """
        image_tensor = image_tensor.to(self.device)
        pil_images = [to_pil_image(img.cpu()) for img in image_tensor]  # Convert to PIL
        preprocessed = torch.stack([self.preprocess(img) for img in pil_images])
        preprocessed = preprocessed.to(self.device)

        with torch.no_grad():
            image_emb = self.model.encode_image(preprocessed)
        return image_emb  # [B, D]

    def forward_text(self, text_tensor):
        if isinstance(text_tensor, torch.Tensor):
            text_tensor = ["" for _ in range(text_tensor.size(0))]  # dummy fallback
        tokens = self.tokenizer(text_tensor).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)
        return text_emb  # [B, D]
