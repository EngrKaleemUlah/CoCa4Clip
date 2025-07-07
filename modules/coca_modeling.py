import torch
import torch.nn as nn
import numpy as np
from modules.coca_wrapper import CoCaWrapper
from modules.transformer import VisualTransformer


class CLIP4Clip(nn.Module):
    def __init__(self, args, tokenizer=None):
        super(CLIP4Clip, self).__init__()
        self.args = args
        self.sim_header = args.sim_header
        self.embedding_dim = 512  # Adjust to match CoCa output dim if needed

        self.clip = CoCaWrapper(model_name=args.coca_model_name)

        if self.sim_header == "seqTransf":
            self.transformer = VisualTransformer(
                width=self.embedding_dim, layers=2, heads=8, attn_mask=None
            )
        elif self.sim_header == "tightTransf":
            self.transformer = VisualTransformer(
                width=self.embedding_dim, layers=2, heads=8, attn_mask=None
            )

        self.frame_agg = args.frame_agg

    def forward_text(self, text):
        return self.clip.forward_text(text)

    def forward_video(self, video_frames):
        """
        video_frames: [B, T, C, H, W]
        """
        B, T, C, H, W = video_frames.shape
        video_frames = video_frames.view(B * T, C, H, W)
        frame_features = self.clip.forward_image(video_frames)  # [B*T, D]
        frame_features = frame_features.view(B, T, -1)  # [B, T, D]

        if self.sim_header == "meanP":
            video_feat = frame_features.mean(dim=1)
        elif self.sim_header in ["seqTransf", "tightTransf"]:
            video_feat = self.transformer(frame_features)[:, 0, :]
        else:
            raise ValueError("Unknown sim_header: {}".format(self.sim_header))

        return video_feat

    def compute_similarity(self, text_feat, video_feat):
        text_feat = nn.functional.normalize(text_feat, dim=-1)
        video_feat = nn.functional.normalize(video_feat, dim=-1)
        return torch.matmul(text_feat, video_feat.t())

    def forward(self, text, video_frames):
        text_feat = self.forward_text(text)  # [B, D]
        video_feat = self.forward_video(video_frames)  # [B, D]
        sim_matrix = self.compute_similarity(text_feat, video_feat)
        return sim_matrix, text_feat, video_feat
