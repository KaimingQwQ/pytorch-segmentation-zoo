import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# swintransformer 论文链接: https://arxiv.org/abs/2103.14030    
class SwinDeepLab(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(SwinDeepLab, self).__init__()
        
        # ---------------------------------------------------------------------
        # 1. 加载 Swin Transformer Backbone (Swin-Tiny)
        # ---------------------------------------------------------------------

        if pretrained:
            weights = models.Swin_T_Weights.DEFAULT
            print(f" 正在加载 SwinTransformer Tiny 预训练权重...")
        else:
            weights = None
            
        backbone = models.swin_t(weights=weights)

        # ---------------------------------------------------------------------
        # 2. 提取特征层 (关键步骤)
        # ---------------------------------------------------------------------

        self.backbone = create_feature_extractor(backbone, return_nodes={'features.7': 'out'})

        # ---------------------------------------------------------------------
        # 3. 构建 ASPP 解码头
        # ---------------------------------------------------------------------
        self.classifier = DeepLabHead(in_channels=768, num_classes=num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:] # (H, W)

        # 1. Backbone 前向传播

        features = self.backbone(x)
        x_feat = features['out'] 
        # 2. 维度调整 

        x_feat = x_feat.permute(0, 3, 1, 2)

        # 3. ASPP 分类头处理
        x_out = self.classifier(x_feat)
        # 4. 上采样回原图尺寸
        x_out = F.interpolate(x_out, size=input_shape, mode='bilinear', align_corners=False)

        return x_out
