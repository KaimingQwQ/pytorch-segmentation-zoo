import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # 加载预训练权重
        base_model = models.resnet101(weights='DEFAULT')

        self.initial = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        )
        self.maxpool = base_model.maxpool # 1/2 -> 1/4

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2 
        self.layer3 = base_model.layer3 
        self.layer4 = base_model.layer4 
        
        # --- Decoder ---
        self.dec4 = UNetDecoderBlock(in_channels=2048, skip_channels=1024, out_channels=512)

        self.dec3 = UNetDecoderBlock(in_channels=512, skip_channels=512, out_channels=256)

        self.dec2 = UNetDecoderBlock(in_channels=256, skip_channels=256, out_channels=128)

        self.dec1 = UNetDecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 1/2 -> 1/1
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        # --- Encoder Forward ---

        x_stem = self.initial(x)      # [B, 64, 256, 256] (1/2) -> Skip 1
        x_pool = self.maxpool(x_stem) # [B, 64, 128, 128] (1/4)
        
        e1 = self.layer1(x_pool)      # [B, 256, 128, 128] (1/4) -> Skip 2
        e2 = self.layer2(e1)          # [B, 512, 64, 64]   (1/8) -> Skip 3
        e3 = self.layer3(e2)          # [B, 1024, 32, 32]  (1/16)-> Skip 4
        e4 = self.layer4(e3)          # [B, 2048, 16, 16]  (1/32) -> Bridge
        
        # --- Decoder Forward ---
        # 1. Layer 4 -> Layer 3
        d4 = self.dec4(e4, skip=e3)   # [B, 512, 32, 32]
        
        # 2. Layer 3 -> Layer 2
        d3 = self.dec3(d4, skip=e2)   # [B, 256, 64, 64]
        
        # 3. Layer 2 -> Layer 1
        d2 = self.dec2(d3, skip=e1)   # [B, 128, 128, 128]
        
        # 4. Layer 1 -> Stem
        d1 = self.dec1(d2, skip=x_stem) # [B, 64, 256, 256]

        out = self.final(d1)          # [B, n_classes, 512, 512]
        
        return out
