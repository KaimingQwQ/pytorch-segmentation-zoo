import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
# OCR 论文链接: https://arxiv.org/abs/1909.11065

# 内容均为AI生成, 仅供参考
# ==================================================================
# Part 1: OCR 核心模块 (Spatial Gather & Object Attention)
# 论文核心公式实现
# ==================================================================

class SpatialGather_Module(nn.Module):
    """
    [Step 2] 聚合对象特征
    将像素特征根据粗分割概率(Soft Object Regions)聚合成K个物体特征。
    公式: f_k = sum(m_ki * x_i)
    """
    def __init__(self, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """
        feats: (B, C, H, W) - 像素特征
        probs: (B, K, H, W) - 粗分割预测 (Soft Object Regions)
        """
        batch_size, c, h, w = feats.size()
        num_classes = probs.size(1)
        
        # 将空间维度拉平: N = H*W
        feats = feats.view(batch_size, c, -1)           # (B, C, N)
        probs = probs.view(batch_size, num_classes, -1) # (B, K, N)
        
        # Spatial Softmax: 对每个类别，归一化所有像素的权重
        # 对应论文 Eq.4 中的 m_ki
        probs = F.softmax(self.scale * probs, dim=2)
        
        # 矩阵乘法: (B, C, N) @ (B, N, K) -> (B, C, K)
        # 得到 K 个 C 维的物体向量
        ocr_context = torch.bmm(feats, probs.permute(0, 2, 1))
        
        return ocr_context

class ObjectAttentionBlock(nn.Module):
    """
    [Step 3] 对象上下文注意力
    计算像素与物体特征的关系，并加权融合。
    """
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        
        # 变换函数: phi, psi, rho, delta (对应论文 Eq.5, Eq.6)
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU()
        )
        self.f_object = nn.Sequential(
            nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(key_channels),
            nn.ReLU()
        )
        self.f_down = nn.Sequential(
            nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(key_channels),
            nn.ReLU()
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, feats, proxy_feats):
        """
        feats:       (B, C, H, W) - 像素特征
        proxy_feats: (B, C, K)    - 物体特征
        """
        batch_size, h, w = feats.size(0), feats.size(2), feats.size(3)
        
        # 1. Query: 像素特征 (B, N, KeyC)
        query = self.f_pixel(feats).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        
        # 2. Key: 物体特征 (B, KeyC, K)
        key = self.f_object(proxy_feats).view(batch_size, self.key_channels, -1)
        
        # 3. Value: 物体特征 (B, K, KeyC)
        value = self.f_down(proxy_feats).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        
        # 4. Attention: Pixel-Object Relation
        # (B, N, KeyC) @ (B, KeyC, K) -> (B, N, K)
        sim_map = torch.bmm(query, key)
        sim_map = (self.scale * sim_map).softmax(dim=-1) # 在 K 个类别上做 Softmax
        
        # 5. Aggregation
        # (B, N, K) @ (B, K, KeyC) -> (B, N, KeyC)
        context = torch.bmm(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, h, w)
        
        # 6. Fusion
        context = self.f_up(context)
        output = feats + context 
        return output

# ==================================================================
# Part 2: Backbone (ViT with Positional Interpolation)
# ==================================================================

class ViTFeatureExtractor(nn.Module):
    """
    封装 torchvision 的 ViT-B/16，支持不同尺寸输入的特征提取。
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = models.vit_b_16(weights=weights)
        self.embed_dim = self.vit.hidden_dim
        self.class_token = self.vit.class_token
        self.pretrained_grid_size = 14 # 224 / 16 = 14

    def _interpolate_pos_encoding(self, x, h, w):
        pos_embed = self.vit.encoder.pos_embedding
        n_ctx = pos_embed.shape[1] - 1
        
        if n_ctx == h * w:
            return pos_embed
        
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        
        # 变成 2D 进行插值
        patch_pos_embed = patch_pos_embed.reshape(1, self.pretrained_grid_size, self.pretrained_grid_size, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, dim)
        
        return torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.vit.conv_proj(x) 
        h_feat, w_feat = x.shape[2], x.shape[3]
        
        x = x.flatten(2).transpose(1, 2)
        batch_class_token = self.class_token.expand(b, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        pos_embed = self._interpolate_pos_encoding(x, h_feat, w_feat)
        x = x + pos_embed
        
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        
        # 移除 CLS token 并重塑回 (B, C, H, W)
        x = x[:, 1:]
        x = x.permute(0, 2, 1).reshape(b, self.embed_dim, h_feat, w_feat)
        return x

# ==================================================================
# Part 3: Decoder (Simple Progressive Upsampling)
# ==================================================================

class ConvUpsampleBlock(nn.Module):
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
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 将输入上采样 16 倍 (2*2*2*2)
        self.up1 = ConvUpsampleBlock(in_channels, 256) # 1/16 -> 1/8
        self.up2 = ConvUpsampleBlock(256, 256)         # 1/8 -> 1/4
        self.up3 = ConvUpsampleBlock(256, 128)         # 1/4 -> 1/2
        self.up4 = ConvUpsampleBlock(128, 64)          # 1/2 -> 1/1
        self.final_cls = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_cls(x)
        return x

# ==================================================================
# Part 4: 完整模型整合 (ViT + OCR + Decoder)
# ==================================================================

class ViTOCRSegmentation(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        
        # 1. Backbone: ViT-B (Output: 768 dim)
        self.backbone = ViTFeatureExtractor(pretrained=pretrained)
        self.backbone_dim = 768
        self.ocr_dim = 512      # OCR操作的特征维度
        self.key_dim = 256      # Attention内部Key/Query维度

        # 2. 转换层: 768 -> 512
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(self.backbone_dim, self.ocr_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ocr_dim),
            nn.ReLU()
        )

        # 3. Auxiliary Head: 用于预测软对象区域 (粗分割)
        # 对应论文中的 Stage 3 -> Aux Loss
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.ocr_dim, self.ocr_dim, kernel_size=1),
            nn.BatchNorm2d(self.ocr_dim),
            nn.ReLU(),
            nn.Conv2d(self.ocr_dim, num_classes, kernel_size=1)
        )

        # 4. OCR 核心模块
        self.spatial_gather = SpatialGather_Module()
        self.object_attn = ObjectAttentionBlock(in_channels=self.ocr_dim, key_channels=self.key_dim)

        # 5. Decoder: 将 OCR 增强后的特征 (512维) 上采样回原图
        self.decoder = SETRDecoder(in_channels=self.ocr_dim, num_classes=num_classes)
        
    def forward(self, x):
        # x shape: (B, 3, H, W) e.g., (2, 3, 512, 512)
        input_size = x.shape[-2:]
        
        # --- Stage 1: Backbone Feature Extraction ---
        # Output: (B, 768, H/16, W/16) -> (2, 768, 32, 32)
        feats = self.backbone(x)
        
        # --- Stage 2: Projection ---
        # Output: (B, 512, 32, 32)
        feats = self.conv_3x3(feats)
        
        # --- Stage 3: Soft Object Regions (Auxiliary Head) ---
        # Output: (B, num_classes, 32, 32)
        out_aux = self.aux_head(feats)
        
        # --- Stage 4: OCR Module ---
        # 4.1: 计算物体特征 (B, 512, K)
        context = self.spatial_gather(feats, out_aux)
        
        # 4.2: 像素-物体 Attention 融合 (B, 512, 32, 32)
        feats_ocr = self.object_attn(feats, context)
        
        # --- Stage 5: Decoder ---
        # Output: (B, num_classes, 512, 512)
        logits = self.decoder(feats_ocr)
        
        # --- Return ---
        if self.training:
            # 训练时需要返回 Aux Output 计算 Loss，记得上采样回原图大小
            out_aux = F.interpolate(out_aux, size=input_size, mode='bilinear', align_corners=True)
            return logits, out_aux
        else:
            return logits

