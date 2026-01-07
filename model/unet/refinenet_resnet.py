import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# 论文链接: https://arxiv.org/abs/1611.06612
# ==================================================================
# 1. 基础组件
# ==================================================================

class ResidualConvUnit(nn.Module):
    """
    Residual Convolution Unit (RCU)
    论文中的 RCU 模块: 两个 3x3 卷积 + ReLU，带有残差连接
    用于对输入特征进行适应性微调
    """
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            # 原论文 RCU 没有 BN
            nn.BatchNorm2d(features) 
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x 

class MultiResolutionFusion(nn.Module):
    """
    Multi-Resolution Fusion (MRF)
    融合不同分辨率的特征：先通过卷积调整通道数，再上采样到最大尺寸，最后相加
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # 为每个输入特征图定义一个 3x3 卷积来统一通道数
        for in_ch in in_channels_list:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            )

    def forward(self, *inputs):
        # inputs 是一个列表，包含 [高分辨率特征, ..., 低分辨率特征]
        assert len(inputs) == len(self.convs) 
        
        # 1. 卷积调整通道
        feats = [conv(x) for conv, x in zip(self.convs, inputs)]
        
        # 2. 上采样并融合
        # 以第一个特征图（分辨率最高）的尺寸为目标
        target_h, target_w = feats[0].shape[2], feats[0].shape[3]
        
        sum_feat = feats[0]
        for i in range(1, len(feats)):
            # 双线性插值上采样
            upsampled = F.interpolate(feats[i], size=(target_h, target_w), mode='bilinear', align_corners=True)
            sum_feat = sum_feat + upsampled
            
        return sum_feat

class ChainedResidualPooling(nn.Module):
    """
    Chained Residual Pooling (CRP)
    链式残差池化：通过多次 MaxPool (5x5) 捕捉大感受野信息
    """
    def __init__(self, features, n_stages=2):
        super().__init__()
        self.n_stages = n_stages
        self.relu = nn.ReLU(inplace=False)
        self.pools = nn.ModuleList()
        
        for _ in range(n_stages):
            self.pools.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features)
                )
            )

    def forward(self, x):
        x = self.relu(x)
        path = x
        
        for pool in self.pools:
            path = pool(path)
            x = x + path # Residual connection
            
        return x

class RefineBlock(nn.Module):
    """
    RefineNet Block (单个精炼模块)
    结构: RCU -> MRF -> RCU -> CRP -> RCU
    """
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        # 1. RCU (对每个输入特征都做)
        self.rcus_in = nn.ModuleList([
            nn.Sequential(
                ResidualConvUnit(ch), 
                ResidualConvUnit(ch)
            ) for ch in in_channels_list
        ])
        
        # 2. MRF (融合)
        self.mrf = MultiResolutionFusion(in_channels_list, out_channels)
        
        # 3. CRP (捕捉上下文)
        self.crp = ChainedResidualPooling(out_channels)
        
        # 4. Final RCU (输出微调)
        self.output_conv = ResidualConvUnit(out_channels)

    def forward(self, *inputs):
        # 1. RCU
        feats = [rcu(x) for rcu, x in zip(self.rcus_in, inputs)]
        
        # 2. MRF
        fused = self.mrf(*feats)
        
        # 3. CRP
        pooled = self.crp(fused)
        
        # 4. Output RCU
        out = self.output_conv(pooled)
        
        return out

# ==================================================================
# 2. RefineNet 主模型 
# ==================================================================

class RefineNetResNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        
        print(" 正在构建 ResNet50 Backbone...")
        
        # 1. Backbone: ResNet
        if pretrained:
            resnet = models.resnet152(weights='DEFAULT')
        else:
            resnet = models.resnet152(weights=None)
            
        # 拆解 ResNet 
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        ) # H/4, 64
        self.layer1 = resnet.layer1 # H/4, 256
        self.layer2 = resnet.layer2 # H/8, 512
        self.layer3 = resnet.layer3 # H/16, 1024
        self.layer4 = resnet.layer4 # H/32, 2048
        
        del resnet

        # 2. RefineNet Blocks 
        # 我们定义 4 个 RefineBlock，分别融合不同层级的特征
        # 通道数通常设为 256 或 512
        self.refine_dim = 256
        
        # RefineBlock 4: 只处理 Layer4 (最深层)
        # Input: [Layer4] -> Output: Refine4
        self.refine4 = RefineBlock([2048], self.refine_dim)
        
        # RefineBlock 3: 融合 Layer3 和 Refine4
        # Input: [Layer3, Refine4] -> Output: Refine3
        self.refine3 = RefineBlock([1024, self.refine_dim], self.refine_dim)
        
        # RefineBlock 2: 融合 Layer2 和 Refine3
        # Input: [Layer2, Refine3] -> Output: Refine2
        self.refine2 = RefineBlock([512, self.refine_dim], self.refine_dim)
        
        # RefineBlock 1: 融合 Layer1 和 Refine2
        # Input: [Layer1, Refine2] -> Output: Refine1 (分辨率 H/4)
        self.refine1 = RefineBlock([256, self.refine_dim], self.refine_dim)
        
        # 3. Final Prediction Head
        # Refine1 输出是 H/4，接两个 RCU 再预测
        self.final_rcu = nn.Sequential(
            ResidualConvUnit(self.refine_dim),
            ResidualConvUnit(self.refine_dim)
        )
        self.clf = nn.Conv2d(self.refine_dim, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:] 
        
        # --- Encoder  ---
        l0 = self.layer0(x) # H/4
        l1 = self.layer1(l0) # H/4, 256
        l2 = self.layer2(l1) # H/8, 512
        l3 = self.layer3(l2) # H/16, 1024
        l4 = self.layer4(l3) # H/32, 2048
        
        # --- Decoder ---
        
        # Path 4: 最深层 (High-level)
        # 输入必须是列表形式
        path4 = self.refine4(l4) # -> H/32, 256
        
        # Path 3: 融合 l3 和 path4
        path3 = self.refine3(l3, path4) # -> H/16, 256
        
        # Path 2: 融合 l2 和 path3
        path2 = self.refine2(l2, path3) # -> H/8, 256
        
        # Path 1: 融合 l1 和 path2
        path1 = self.refine1(l1, path2) # -> H/4, 256
        
        # --- Final Prediction ---
        out = self.final_rcu(path1)
        out = self.clf(out) # -> H/4, n_classes
        
        # 4x Upsampling to original resolution
        return F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
