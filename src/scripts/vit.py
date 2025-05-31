import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16


class ViT(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, image_size=128):
        super().__init__()
        # Pre-trained ViT (modify to accept 3-channel inputs)
        self.backbone = vit_b_16(pretrained=False)
        self.backbone.conv_proj = nn.Conv2d(in_channels, 768, kernel_size=16, stride=16)

        self.mlp_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, image_size * image_size * out_channels)
        )

        self.out_channels = out_channels
        self.image_size = image_size

    def forward(self, x):
        features = self.backbone(x)
        out = self.mlp_head(features)
        return out.view(-1, self.out_channels, self.image_size, self.image_size)