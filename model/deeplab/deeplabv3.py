import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
#论文链接: https://arxiv.org/abs/1706.05587
class DeepLabV3(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(DeepLabV3, self).__init__()
        
        # 1. 加载官方模型
        if pretrained:
            print(" 正在加载 DeepLabV3 ResNet50 (COCO 预训练)...")
            full_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        else:
            full_model = models.segmentation.deeplabv3_resnet50(pretrained=False)
            
        # 2. 提取 Backbone (Dilated ResNet)
        self.backbone = full_model.backbone   
            
        # 3. 提取 ASPP 分类头 (Classifier)
        self.classifier = full_model.classifier      
          
        # 4. 修改分类头的最后一层 
        in_channels = self.classifier[-1].in_channels    
        self.classifier[-1] = nn.Conv2d(in_channels, n_classes, kernel_size=1)
        
        # 5. 处理辅助分类头 (Aux Classifier) - 可选

        self.aux_classifier = full_model.aux_classifier
        if self.aux_classifier is not None:
            aux_in = self.aux_classifier[-1].in_channels
            self.aux_classifier[-1] = nn.Conv2d(aux_in, n_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:] 
        
        # 1. Backbone 前向传播
        features = self.backbone(x)
        
        result = {}
        
        # 2. 主分支 (Layer4 -> ASPP -> Out)
        x_main = features['out']
        x_main = self.classifier(x_main)
        x_main = F.interpolate(x_main, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x_main
        
        # 3. 辅助分支 
        if self.training and self.aux_classifier is not None:
            x_aux = features['aux']
            x_aux = self.aux_classifier(x_aux)
            x_aux = F.interpolate(x_aux, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = x_aux
            
            # 训练时返回字典，方便计算两个 Loss
            return result

        return result['out']
