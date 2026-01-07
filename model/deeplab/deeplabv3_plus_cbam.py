import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 论文链接: https://arxiv.org/abs/1807.06521

# =====================================================================
# 1. 定义 CBAM 模块
# =====================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享 MLP (多层感知机)
        # 注意：使用 1x1 卷积代替全连接层，可以处理任意尺寸输入，且参数共享
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 相加后 Sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # 2通道输入：一个是 max_pool 结果，一个是 avg_pool 结果
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度(dim=1)上做平均和最大操作
        # (B, C, H, W) -> (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接 -> (B, 2, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 卷积 -> Sigmoid
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 1. 先做通道注意力
        out = x * self.ca(x)
        # 2. 再做空间注意力
        result = out * self.sa(out)
        return result

# =====================================================================
# 2. 集成 CBAM 的 DeepLabv3Plus
# =====================================================================

class DeepLabV3PlusCBAM(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(DeepLabV3PlusCBAM, self).__init__()
        
        # 1. 加载 Backbone (ResNet50)
        base_model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        self.backbone = base_model.backbone
        # 提取 layer1 (低级特征) 和 layer4 (高级特征)
        self.backbone.return_layers = {'layer1': 'layer1', 'layer4': 'layer4'}

        # 2. ASPP 模块 (输出通道固定为 256)
        self.aspp = base_model.classifier[0]

        # 3. 插入 CBAM 模块
        # ASPP 输出是 256 通道，所以 CBAM 输入也是 256
        self.cbam = CBAM(in_planes=256, ratio=16, kernel_size=7)

        # 4. Decoder 部分
        # 4.1 低级特征降维 (256 -> 48)
        self.low_level_project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 4.2 最终分类头
        self.final_classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False), # 304 = 256(ASPP) + 48(Low)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:] # (H, W)

        # ---------------------
        # Encoder
        # ---------------------
        features = self.backbone(x)
        low_level_feat = features['layer1'] # (B, 256, H/4, W/4)
        x_feat = features['layer4']         # (B, 2048, H/16, W/16)

        # ---------------------
        # ASPP + CBAM
        # ---------------------
        x_aspp = self.aspp(x_feat) # (B, 256, H/16, W/16)
        # 应用 CBAM 注意力机制
        x_aspp = self.cbam(x_aspp) # 维度不变，但特征经过了重校准

        # ---------------------
        # Decoder
        # ---------------------

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