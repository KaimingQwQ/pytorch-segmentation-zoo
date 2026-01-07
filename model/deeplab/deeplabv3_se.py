import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# 论文链接: https://arxiv.org/abs/1709.01507

# ---------------------------------------------------------------------
# 1. 定义 SE 模块 (Squeeze-and-Excitation)
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

class DeepLabV3SE(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super(DeepLabV3SE, self).__init__()
        
        # 1. 加载 torchvision 的 DeepLabV3 (ResNet50) 模型
        base_model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        
        # ---------------------------------------------------------------------
        # 2. 修改 Backbone 输出配置
        # ---------------------------------------------------------------------
        self.backbone = base_model.backbone
        
        # DeepLabV3 只需要 High-level 特征 (Layer4)，不需要 Layer1
        self.backbone.return_layers = {'layer4': 'layer4'}

        # ---------------------------------------------------------------------
        # 3. 复用 ASPP 模块
        # ---------------------------------------------------------------------
        # base_model.classifier[0] 是 ASPP 模块
        self.aspp = base_model.classifier[0]

        # ---------------------------------------------------------------------
        # 4. 插入 SE 模块 
        # ---------------------------------------------------------------------

        self.se_block = SEBlock(in_channels=256, reduction=16)

        # ---------------------------------------------------------------------
        # 5. 构建最终分类头 (Classifier)
        # ---------------------------------------------------------------------

        self.final_classifier = nn.Sequential(
            # 3x3 卷积 (ASPP out 256 -> 256)
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:] # (H, W)
        # 1. Backbone 前向传播
        features = self.backbone(x)
        x_feat = features['layer4']      

        # 2. ASPP 处理
        x = self.aspp(x_feat)        
        print(x.shape)
        # 3. SE 模块增强
        x = self.se_block(x) 
        # 4. 最终分类卷积
        x = self.final_classifier(x)  

        # 5. 上采样恢复
        x_out = F.interpolate(
            x, 
            size=input_shape, 
            mode='bilinear', 
            align_corners=False
        )

        return x_out
