import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class RCU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCU, self).__init__()
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        residual = self.proj(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual

class MRF(nn.Module):
    def __init__(self, in_channels_list, out_channels):  
        super(MRF, self).__init__()
        for i, in_channels in enumerate(in_channels_list):
            setattr(self, f'rcu_{i}', RCU(in_channels, out_channels))
        self.out_channels = out_channels

    def forward(self, x_list):
        rcu_outs = []
        for i, x in enumerate(x_list):
            rcu = getattr(self, f'rcu_{i}')
            rcu_outs.append(rcu(x))
        
        # 动态寻找最大尺寸
        max_h = max([feat.shape[2] for feat in rcu_outs])
        max_w = max([feat.shape[3] for feat in rcu_outs])
        
        upsampled_feats = []
        for feat in rcu_outs:
            if feat.shape[2] != max_h or feat.shape[3] != max_w:
                upsampled_feat = F.interpolate(feat, size=(max_h, max_w), mode='bilinear', align_corners=True) # 建议 align_corners=True
            else:
                upsampled_feat = feat
            upsampled_feats.append(upsampled_feat)
        
        fused_feat = torch.stack(upsampled_feats, dim=0).sum(dim=0)
        return fused_feat

class CRPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, n_stages):
        super(CRPPooling, self).__init__()
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        for i in range(n_stages):
            # 为了通用性，虽然这里 input_c=out_channels，但保留逻辑
            input_c = in_channels if i == 0 else out_channels
            setattr(self, f'conv_{i}', nn.Conv2d(input_c, out_channels, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        top = x
        out = x 
        for i in range(self.n_stages):
            top = self.maxpool(top)
            conv = getattr(self, f'conv_{i}')
            top = conv(top)
            out = out + top
        return out

class RefineNetBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels, n_crp_stages=4):
        super(RefineNetBlock, self).__init__()
        self.mrf = MRF(in_channels_list, out_channels)
        self.crp = CRPPooling(out_channels, out_channels, n_crp_stages)
        self.rcu = RCU(out_channels, out_channels)

    def forward(self, x_list):
        x = self.mrf(x_list)
        x = self.crp(x)
        x = self.rcu(x)
        return x

class RefineNetMultiScale(nn.Module):
    def __init__(self, num_classes=21):
        super(RefineNetMultiScale, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        

        self.init=nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.refinenet4 = RefineNetBlock([2048, 1024, 2048], 512, 4)
        self.refinenet3 = RefineNetBlock([512, 512, 1024], 256, 4)
        self.refinenet2 = RefineNetBlock([256, 512], 256, 4)
        self.refinenet1 = RefineNetBlock([256, 256], 256, 4)
        
        self.final_rcu = RCU(256, 256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def extract_features(self, x):
        x = self.init(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return l1, l2, l3, l4

    def forward(self, x):
        # 多尺度输入
        X0_6 = F.interpolate(x, scale_factor=0.6, mode='bilinear', align_corners=True)
        _, X0_6l2, X0_6l3, X0_6l4 = self.extract_features(X0_6)
        
        X1_2 = F.interpolate(x, scale_factor=1.2, mode='bilinear', align_corners=True)
        X1_2l1, X1_2l2, X1_2l3, X1_2l4 = self.extract_features(X1_2)
        
        # 级联 RefineNet
        r4 = self.refinenet4([X0_6l4, X0_6l3, X1_2l4]) # [2048, 1024, 2048] -> 512
        r3 = self.refinenet3([X0_6l2, r4, X1_2l3])     # [512, 512, 1024] -> 256
        r2 = self.refinenet2([r3, X1_2l2])             # [256, 512] -> 256
        r1 = self.refinenet1([r2, X1_2l1])             # [256, 256] -> 256
        
        out = self.final_rcu(r1)
        out = self.classifier(out)

        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out