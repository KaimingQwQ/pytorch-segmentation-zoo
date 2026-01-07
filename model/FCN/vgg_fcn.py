import torchvision
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
#vgg论文链接: https://arxiv.org/abs/1409.1556
class VGGFCN(nn.Module):
    def __init__(self, n_classes):
        super(VGGFCN, self).__init__()
        print(" 正在加载 VGG16_BN ...")

        self.fullmodel = torchvision.models.vgg16_bn(pretrained=False)
        self.features_block1 = self.fullmodel.features[:24] 
        self.features_block2 = self.fullmodel.features[24:34]
        self.features_block3 = self.fullmodel.features[34:]
        self.score_pool5 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, n_classes, kernel_size=1)

        self.up2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


    def forward(self, x):
        input_size = x.shape[-2:]
        # --- 前向传播提取特征 ---
            
        pool3 = self.features_block1(x) 
        pool4 = self.features_block2(pool3)
        pool5 = self.features_block3(pool4)

        # --- 融合 ---
        
        # === 融合阶段 1: Pool5 + Pool4 ===
        score5 = self.score_pool5(pool5)
        up_score5 = self.up2x(score5) # 32s -> 16s
        
        score4 = self.score_pool4(pool4)
        
        if up_score5.shape[-2:] != score4.shape[-2:]:
            up_score5 = F.interpolate(up_score5, size=score4.shape[-2:], mode='bilinear', align_corners=True)
            
        fuse_pool4 = up_score5 + score4
        
        # === 融合阶段 2: Fuse4 + Pool3 ===
        up_fuse4 = self.up2x(fuse_pool4) # 16s -> 8s
        score3 = self.score_pool3(pool3)

        if up_fuse4.shape[-2:] != score3.shape[-2:]:
            up_fuse4 = F.interpolate(up_fuse4, size=score3.shape[-2:], mode='bilinear', align_corners=True)
            
        fuse_pool3 = up_fuse4 + score3

        # === 最终输出 ===
        # 8s -> 1s 
        out = self.up8x(fuse_pool3)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out
