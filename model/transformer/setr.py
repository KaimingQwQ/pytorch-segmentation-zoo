import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
# 论文链接: https://arxiv.org/abs/2012.15840

# AI生成代码，仅供参考，需验证正确性！
# ==================================================================
# 1. Backbone: Vision Transformer (with Positional Encoding Interpolation)
# ==================================================================

class ViTFeatureExtractor(nn.Module):
    """
    封装 torchvision 的 ViT-B/16，使其能输出 2D 特征图。
    关键功能: 处理位置编码插值，以支持不同尺寸的输入 (如 512x512)。
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载 ImageNet 预训练权重
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = models.vit_b_16(weights=weights)
        
        # 获取 Patch Size 和 Embed Dim (768)
        self.patch_size = self.vit.patch_size
        self.embed_dim = self.vit.hidden_dim
        
        # 移除原有的分类头，只保留 Encoder 部分
        # ViT 的核心流程: class_token + patch_embeddings -> encoder -> ln
        self.class_token = self.vit.class_token
        self.encoder = self.vit.encoder
        # 兼容不同版本的 torchvision ViT 结构
        if hasattr(self.vit.encoder, 'ln'):
             self.encoder_norm = self.vit.encoder.ln 
        elif hasattr(self.vit.encoder.layers[-1], 'ln'):
             self.encoder_norm = self.vit.encoder.layers[-1].ln
        else:
             self.encoder_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # 原始预训练时的 patch grid size (224/16 = 14)
        self.pretrained_grid_size = 14 

    def _interpolate_pos_encoding(self, x, h, w):
        # 1. 获取预训练的位置编码
        pos_embed = self.vit.encoder.pos_embedding # (1, 197, 768)
        
        # 2. 如果尺寸刚好匹配，直接返回，不用插值
        n_ctx = pos_embed.shape[1] - 1
        if n_ctx == h * w:
            return pos_embed

        # 3. 分离 Class Token 和 Patch Tokens
        class_pos_embed = pos_embed[:, 0]      # (1, 768)
        patch_pos_embed = pos_embed[:, 1:]     # (1, 196, 768)
        
        dim = x.shape[-1]
        
        # 4. Reshape 成 (1, 768, 14, 14) 准备插值
        w0 = h0 = int(math.sqrt(n_ctx)) # 原始 grid size (通常是 14)
        
        # (1, 196, 768) -> (1, 14, 14, 768) -> (1, 768, 14, 14)
        patch_pos_embed = patch_pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)
        
        # 5. 双线性/双三次插值到 (h, w)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.float(), 
            size=(h, w), 
            mode='bicubic', 
            align_corners=False
        ).to(x.dtype) 
        
        # 6. 变回序列: (1, 768, h, w) -> (1, h*w, 768)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        # 7. 拼接回 Class Token
        return torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 1. 卷积 Patch Embedding
        # (B, 768, H/16, W/16)
        x = self.vit.conv_proj(x) 
        
        h_feat, w_feat = x.shape[2], x.shape[3]
        
        # (B, 768, N) -> (B, N, 768)
        x = x.flatten(2).transpose(1, 2)
        
        # 2. 添加 Class Token
        batch_class_token = self.class_token.expand(b, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # 3. 添加动态插值后的位置编码
        pos_embed = self._interpolate_pos_encoding(x, h_feat, w_feat)
        x = x + pos_embed
        
        # 4. Transformer Encoder Layers
        x = self.vit.encoder.layers(x)
        x = self.encoder_norm(x)
        
        # 5. 移除 Class Token 并重塑回 2D
        # x: (B, 1 + H*W, 768) -> (B, H*W, 768)
        x = x[:, 1:]
        # (B, H*W, 768) -> (B, 768, H, W)
        x = x.permute(0, 2, 1).reshape(b, self.embed_dim, h_feat, w_feat)
        
        return x

# ==================================================================
# 2. Decoder: PUP Head (Progressive Upsampling)
# ==================================================================

class ConvUpsampleBlock(nn.Module):
    """
    基本的上采样块: Conv -> BN -> ReLU -> 2x Upsample
    """
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return x

class SETRDecoder(nn.Module):
    """
    SETR 的解码器: 将 ViT 的特征图 (1/16) 逐步上采样回 (1/1)
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.up1 = ConvUpsampleBlock(in_channels, 256) # 1/16 -> 1/8
        self.up2 = ConvUpsampleBlock(256, 256)         # 1/8 -> 1/4
        self.up3 = ConvUpsampleBlock(256, 128)         # 1/4 -> 1/2
        self.up4 = ConvUpsampleBlock(128, 64)      
        
        self.final_cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_cls(x)
        return x

# ==================================================================
# 3. 主模型: SETR (ViT + Decoder)
# ==================================================================

class SETR(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        
        # 1. Transformer Backbone
        self.backbone = ViTFeatureExtractor(pretrained=pretrained)
        
        # 2. CNN Decoder
        # ViT-B 输出 768 维
        self.decoder = SETRDecoder(in_channels=768, num_classes=num_classes)
        
    def forward(self, x):
        # x: (B, 3, 512, 512)
        input_shape = x.shape[-2:]
        
        # Backbone 提取特征: (B, 768, 32, 32)
        features = self.backbone(x)
        
        # Decoder 上采样预测: (B, 21, 512, 512)
        logits = self.decoder(features)
        
        if logits.shape[-2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=True)
            
        return logits