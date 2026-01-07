import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------------------
# 1. SE 模块定义
# ---------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: 全连接层 -> ReLU -> 全连接层 -> Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)

        return x * y

# ---------------------------------------------------------------------
# 2. 修改后的 DeepLabV3Plus
# ---------------------------------------------------------------------
class DeepLabV3PlusSE(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(DeepLabV3PlusSE, self).__init__()
        
        # 加载 torchvision 的 DeepLabV3 (ResNet50) 模型
        base_model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)

        self.backbone = base_model.backbone
        self.backbone.return_layers = {'layer1': 'layer1', 'layer4': 'layer4'}

        self.aspp = base_model.classifier[0]

        # 在 ASPP 后面加入 SE Block
        self.se = SEBlock(in_channels=256, reduction=16)
        # ===============================================

        # 构建 Decoder
        # 4.1 低级特征投影层 (1x1 Conv)
        self.low_level_project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 4.2 最终分类头
        self.final_classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:] # (H, W)

        # 1. Backbone 前向传播
        features = self.backbone(x)
        low_level_feat = features['layer1'] # (B, 256, H/4, W/4)
        x_feat = features['layer4']         # (B, 2048, H/16, W/16)

        # 2. ASPP 处理高级特征
        x_aspp = self.aspp(x_feat) # Output: (B, 256, H/16, W/16)

        x_aspp = self.se(x_aspp)

        # 3. Decoder 融合
        x_aspp_up = F.interpolate(
            x_aspp, 
            size=low_level_feat.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )

        low_level_feat = self.low_level_project(low_level_feat)

        x_cat = torch.cat([x_aspp_up, low_level_feat], dim=1) 

        x_out = self.final_classifier(x_cat)

        x_out = F.interpolate(
            x_out, 
            size=input_shape, 
            mode='bilinear', 
            align_corners=False
        )

        return x_out
