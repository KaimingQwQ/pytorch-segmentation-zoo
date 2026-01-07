import torch.nn as nn
import torch.nn.functional as F
import torchvision
# Resnet论文链接: https://arxiv.org/abs/1512.03385
# FCN论文链接: https://arxiv.org/abs/1411.4038
class ResNetFCN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        

        print("加载Backbone")
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        resnet = torchvision.models.resnet50(weights=weights)
        
        # 拆解 Backbone
        self.initial = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
            resnet.layer1
        )
        
        # Layers
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 降维投影层

        self.score_32s = nn.Conv2d(2048, n_classes, kernel_size=1)
        self.score_16s = nn.Conv2d(1024, n_classes, kernel_size=1)
        self.score_8s  = nn.Conv2d(512, n_classes, kernel_size=1)
        
        # 上采样层 
        self.up2x_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2x_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up8x   = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


    def forward(self, x):

        x = self.initial(x)
        c2 = self.layer2(x) 
        c3 = self.layer3(c2) 
        c4 = self.layer4(c3) 
        
        score_32 = self.score_32s(c4)
        up_32 = self.up2x_1(score_32)
        score_16 = self.score_16s(c3)

        if up_32.shape != score_16.shape:
             up_32 = F.interpolate(up_32, size=score_16.shape[2:], mode='bilinear', align_corners=True)
        
        fuse_16 = up_32 + score_16
        up_16 = self.up2x_2(fuse_16)
        
        score_8 = self.score_8s(c2)
        if up_16.shape != score_8.shape:
             up_16 = F.interpolate(up_16, size=score_8.shape[2:], mode='bilinear', align_corners=True)

        fuse_8 = up_16 + score_8
        out = self.up8x(fuse_8)
        
        return out

class RefineBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResNetFCNRefined(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # 1. 加载预训练的 ResNet50 Backbone
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        resnet = torchvision.models.resnet50(weights=weights)
        
        # 2. 拆解 Backbone
        self.initial = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
            resnet.layer1
        )
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 3. 降维投影层
        # ------------------------------------------------------
        self.score_32s = nn.Conv2d(2048, n_classes, kernel_size=1)
        self.score_16s = nn.Conv2d(1024, n_classes, kernel_size=1)
        self.score_8s  = nn.Conv2d(512, n_classes, kernel_size=1)
        
        # 4. 特征平滑层 
        # ------------------------------------------------------
        self.refine_16s = RefineBlock(n_classes)
        self.refine_8s  = RefineBlock(n_classes)
        
        # 5. 上采样层
        # ------------------------------------------------------
        self.up2x_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2x_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up8x   = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.initial(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        # 1. Stride 32 -> 16
        score_32 = self.score_32s(c4)
        up_32 = self.up2x_1(score_32)
        score_16 = self.score_16s(c3)
        if up_32.shape[-2:] != score_16.shape[-2:]:
             up_32 = F.interpolate(up_32, size=score_16.shape[2:], mode='bilinear', align_corners=True)
        fuse_16 = up_32 + score_16
        fuse_16 = self.refine_16s(fuse_16)
         
        # 2. Stride 16 -> 8
        up_16 = self.up2x_2(fuse_16)
        score_8 = self.score_8s(c2)
        if up_16.shape[-2:] != score_8.shape[-2:]:
             up_16 = F.interpolate(up_16, size=score_8.shape[2:], mode='bilinear', align_corners=True)
        fuse_8 = up_16 + score_8
        fuse_8 = self.refine_8s(fuse_8)
        
        
        # 3. 最终还原
        out = self.up8x(fuse_8)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
            
        return out
    
class ResidualRefineBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return self.relu(out)
