import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# 论文链接: https://arxiv.org/abs/1802.02611
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        
        # 1. 加载 torchvision 的 DeepLabV3 (ResNet50) 模型
        base_model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        
        # ---------------------------------------------------------------------
        # 2. 修改 Backbone 输出配置 
        # ---------------------------------------------------------------------
        self.backbone = base_model.backbone
        
        # 这里的字典决定了 features = self.backbone(x) 返回的字典里的 Key 是什么
        # 修改为: {'内部层名': '外部调用名'} -> {'layer1': 'layer1', 'layer4': 'layer4'}
        self.backbone.return_layers = {'layer1': 'layer1', 'layer4': 'layer4'}

        # ---------------------------------------------------------------------
        # 3. 复用 ASPP 模块
        # ---------------------------------------------------------------------
        self.aspp = base_model.classifier[0]

        # ---------------------------------------------------------------------
        # 4. 构建 Decoder
        # ---------------------------------------------------------------------
        # 4.1 低级特征投影层 (1x1 Conv)
        # ResNet layer1 输出通道是 256，降维到 48
        self.low_level_project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 4.2 最终分类头
        self.final_classifier = nn.Sequential(
            # 3x3 卷积融合特征 (输入 256+48=304 -> 输出 256)
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 映射到类别数
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:] # (H, W)

        # 1. Backbone 前向传播
        features = self.backbone(x)
        
        low_level_feat = features['layer1'] # shape: (B, 256, H/4, W/4)
        x_feat = features['layer4']         # shape: (B, 2048, H/16 or H/8, W/16 or W/8)

        x_aspp = self.aspp(x_feat)

        # 3. Decoder 融合
        x_aspp_up = F.interpolate(
            x_aspp, 
            size=low_level_feat.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )

        # 处理低级特征 
        low_level_feat = self.low_level_project(low_level_feat)
        
        # 拼接
        x_cat = torch.cat([x_aspp_up, low_level_feat], dim=1) 

        x_out = self.final_classifier(x_cat)

        x_out = F.interpolate(
            x_out, 
            size=input_shape, 
            mode='bilinear', 
            align_corners=False
        )

        return x_out
