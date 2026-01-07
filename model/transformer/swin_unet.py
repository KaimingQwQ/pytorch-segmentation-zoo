import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

def conv_bn_relu(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    """
    上采样 + 与 skip 特征拼接 + 两层卷积融合
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = conv_bn_relu(in_ch + skip_ch, out_ch)
        self.conv2 = conv_bn_relu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SwinUNet(nn.Module):
    """
    Swin-T backbone + UNet-like decoder
    - 用 stage1/2/3/4 多尺度特征做跳连
    - 逐级上采样恢复空间细节
    """
    def __init__(self, num_classes=21, pretrained=True, decoder_channels=(256, 128, 64, 32)):
        super().__init__()

        if pretrained:
            weights = models.Swin_T_Weights.DEFAULT
            print(" 正在加载 SwinTransformer  预训练权重...")
        else:
            weights = None

        backbone = models.swin_t(weights=weights)

        # 0 PatchPartition
        # 1 Stage1  -> stride 4,  channels 96
        # 2 PatchMerging
        # 3 Stage2  -> stride 8,  channels 192
        # 4 PatchMerging
        # 5 Stage3  -> stride 16, channels 384
        # 6 PatchMerging
        # 7 Stage4  -> stride 32, channels 768
        self.backbone1 = create_feature_extractor(
            backbone,
            return_nodes={
                "features.1": "s1",  # (N, H/4,  W/4,  96)
                "features.3": "s2",  # (N, H/8,  W/8,  192)
                "features.5": "s3",  # (N, H/16, W/16, 384)
                "features.7": "s4",  # (N, H/32, W/32, 768)
            },
        )

        # 把 channel-last -> channel-first 之后，用 1x1 conv 统一到解码器通道数，降低解码器计算量
        c1, c2, c3, c4 = 96, 192, 384, 768
        d4, d3, d2, d1 = decoder_channels  # 从深到浅：256,128,64,32

        self.proj4 = nn.Conv2d(c4, d4, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(c3, d3, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(c2, d2, kernel_size=1, bias=False)
        self.proj1 = nn.Conv2d(c1, d1, kernel_size=1, bias=False)

        # 解码器：s4 -> s3 -> s2 -> s1
        self.up43 = UpBlock(in_ch=d4, skip_ch=d3, out_ch=d3)
        self.up32 = UpBlock(in_ch=d3, skip_ch=d2, out_ch=d2)
        self.up21 = UpBlock(in_ch=d2, skip_ch=d1, out_ch=d1)

        # 最终分类头：输出到 num_classes
        self.head = nn.Conv2d(d1, num_classes, kernel_size=1)

    @staticmethod
    def _to_nchw(x_nhwc: torch.Tensor) -> torch.Tensor:
        # (N, H, W, C) -> (N, C, H, W)
        return x_nhwc.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        input_shape = x.shape[-2:]

        feats = self.backbone1(x)
        s1 = self._to_nchw(feats["s1"])  # stride 4

        s2 = self._to_nchw(feats["s2"])  # stride 8

        s3 = self._to_nchw(feats["s3"])  # stride 16

        s4 = self._to_nchw(feats["s4"])  # stride 32

        # 1x1 投影
        p4 = self.proj4(s4)
        p3 = self.proj3(s3)
        p2 = self.proj2(s2)
        p1 = self.proj1(s1)

        # UNet 解码
        x = self.up43(p4, p3)
        x = self.up32(x, p2)
        x = self.up21(x, p1)   

        logits = self.head(x) 
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        return logits

