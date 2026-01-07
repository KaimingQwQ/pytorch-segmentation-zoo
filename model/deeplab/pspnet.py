import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# 论文链接: https://arxiv.org/abs/1612.01105
class PyramidPoolingModule(nn.Module):
    """
    PSPNet 的核心模块: Pyramid Pooling Module (PPM)
    功能: 将输入特征图进行不同尺度的池化 (1x1, 2x2, 3x3, 6x6)，
          然后上采样回原尺寸并与原始特征图拼接。
    """
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, size) for size in sizes
        ])
        
    def _make_stage(self, in_channels, size):
        # 按照论文，每个金字塔层的输出通道通常减少为原来的 1/N (N=4)
        # 这里 ResNet50 layer4 输出 2048，除以 4 得到 512
        out_channels = in_channels // 4
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        ppm_outs = [x]
        
        for stage in self.stages:
            # 1. 池化 + 1x1 卷积
            feat = stage(x)
            # 2. 上采样回原特征图尺寸
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            ppm_outs.append(feat)
            
        # 3. 拼接: 原始特征 + 4个金字塔特征
        return torch.cat(ppm_outs, dim=1)

class PSPNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(PSPNet, self).__init__()
        
        # ---------------------------------------------------------------------
        # 1. 加载 Backbone
        # ---------------------------------------------------------------------
        base_model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        self.backbone = base_model.backbone
        
        # PSPNet 主要只需要最高层的特征 (Layer4)
        self.backbone.return_layers = {'layer4': 'layer4'}

        # ---------------------------------------------------------------------
        # 2. 构建 PPM 模块
        # ---------------------------------------------------------------------
        self.ppm = PyramidPoolingModule(in_channels=2048, sizes=(1, 2, 3, 6))

        # ---------------------------------------------------------------------
        # 3. 最终分类头 (Decoder)
        # ---------------------------------------------------------------------
        self.final_classifier = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
            
            # 最终分类映射
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:] # (H, W)
        # 1. Backbone 前向传播
        features = self.backbone(x)
        x_feat = features['layer4'] 

        # 2. PPM 模块处理 
        x_ppm = self.ppm(x_feat)   

        # 3. 分类头
        x_out = self.final_classifier(x_ppm)

        x_out = F.interpolate(
            x_out, 
            size=input_shape, 
            mode='bilinear', 
            align_corners=False
        )

        return x_out

