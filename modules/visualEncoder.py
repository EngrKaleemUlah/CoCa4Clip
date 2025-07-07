import torch
import torch.nn as nn
import timm


class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "efficientvit_b3.r288_in1k", output_dim: int = 512):
        super(ViTFeatureExtractor, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(
            "cuda"
        )
        self.model.eval()

        # Global pooling layer (some timm models don’t apply it automatically)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Project to match CLIP output dim if needed
        self.project = None
        num_features = self.model.num_features
        if num_features != output_dim:
            self.project = nn.Linear(num_features, output_dim).to("cuda")

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, video: torch.Tensor, num_frames: int):
        """
        video: [B * T, 3, H, W]
        Returns: [B, T, D]
        """
        B_T, C, H, W = video.shape
        video = video / 255.0 if video.max() > 1 else video
        video = (video - self.mean.to(video.device)) / self.std.to(video.device)

        with torch.no_grad():
            features = self.model.forward_features(video)  # [B*T, C, H, W]
            if features.ndim == 4:
                features = self.pool(features).squeeze(-1).squeeze(-1)  # [B*T, C]

        if self.project:
            features = self.project(features)  # [B*T, D]

        B = B_T // num_frames
        return features.view(B, num_frames, -1).float()  # [B, T, D]
