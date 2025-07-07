import torch
from torch import nn
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"


# class SentenceEncoder(nn.Module):
#     """
#     SentenceEncoder replaces the original text encoder with a Sentence Transformer model.
#     """

#     def __init__(self, model_name="all-mpnet-base-v2", output_dim=512):
#         """
#         Args:
#             model_name (str): Name of the Sentence Transformer model to load.
#             output_dim (int): Dimension to which the Sentence Transformer output will be mapped.
#         """
#         super(SentenceEncoder, self).__init__()
#         self.sentence_transformer = SentenceTransformer(model_name)
#         # Linear layer to project the output to the desired dimension
#         self.linear = nn.Linear(768, output_dim)

#     def forward(self, input_ids):
#         """
#         Forward pass to compute the sentence embeddings.

#         Args:
#             input_ids (torch.Tensor): Tokenized text input for the Sentence Transformer.

#         Returns:
#             torch.Tensor: Transformed sentence embeddings.
#         """
#         # Encode input text using the Sentence Transformer
#         with torch.no_grad():  # Prevent gradients for the Sentence Transformer
#             embeddings = self.sentence_transformer.encode(
#                 # input_ids.cpu().tolist(),  # Convert to list of strings
#                 input_ids,
#                 convert_to_tensor=True,
#                 show_progress_bar=False
#             )

#         embeddings = embeddings.to(device)  # Ensure embeddings are on the same device
#         # Project the embeddings to the desired output dimension
#         transformed_embeddings = self.linear(embeddings)
#         return transformed_embeddings

import open_clip


class SentenceEncoder(nn.Module):
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
        # image_tensor: [B, C, H, W] or [B*T, C, H, W]
        image_tensor = image_tensor.to(self.device)
        image_tensor = torch.stack([self.preprocess(img) for img in image_tensor])
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            image_emb = self.model.encode_image(image_tensor)
        return image_emb  # [B, D]

    def forward_text(self, text_tensor):
        if isinstance(text_tensor, torch.Tensor):
            text_tensor = ["" for _ in range(text_tensor.size(0))]  # dummy fallback
        tokens = self.tokenizer(text_tensor).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(tokens)  # output shape [1,768]
        return text_emb  # [B, D]
