import os
from pathlib import Path

# ==============================================================================
# 1. 模型导入 
# ==============================================================================
from model import (
    # --- Transformer & Hybrid ---
    SwinUNet,
    SETR,
    SwinDeepLab,
    vit_ocr,
    
    # --- DeepLab Series ---
    DeepLabV3Plus,
    DeepLabV3PlusCBAM,
    DeepLabV3PlusSE,
    DeepLabV3SE,
    deeplabv3,
    PSPNet,

    # --- UNet & RefineNet Series ---
    RefineNetMultiScale,
    RefineNetResNet,
    ResNetUNet,

    # --- FCN Series ---
    ResNetFCNRefined,
    ResNetFCN,
    VGGFCN
)

# ==============================================================================
# 2. 路径配置
# ==============================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = Path("~/kaiming/Dataset").expanduser()
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
RESULT_DIR = os.path.join(ROOT_DIR, 'results')

# ==============================================================================
# 3. 数据与加载参数 
# ==============================================================================
img_size = 512
NUM_CLASSES = 21
BATCH_SIZE = 16
NUM_WORKERS = 4

# ==============================================================================
# 4. 基础训练超参数 
# ==============================================================================
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3

# 损失函数设置
LOSS_TYPE = 'ce'       # 可选: 'ce' (仅交叉熵), 'ce_dice' (混合损失)
DICE_WEIGHT = 0.4      # Dice Loss 的权重 (当 LOSS_TYPE='ce_dice' 时生效)

# ==============================================================================
# 5. 模型选择 
# ==============================================================================
# 默认模型实例
MODEL = SwinUNet(NUM_CLASSES)
MODLE = MODEL

# ==============================================================================
# 6. 高级训练策略 
# ==============================================================================
# 学习率预热
WARMUP_EPOCHS = 10

# Lovasz Loss 微调策略
USE_LOVASZ_FINETUNE = False      # 是否在后期开启 Lovasz Loss 进行微调
LOVASZ_SWITCH_EPOCH = 70         # 在第几个 Epoch 切换到 Lovasz 
LOVASZ_WEIGHT = 0.75             # Lovasz Loss 的权重

# ==============================================================================
# 7. 优化与迁移学习 
# ==============================================================================
# Backbone 差异化学习率配置
backbone_keywords = ['backbone', 'initial', 'layer1', 'layer2', 'layer3', 'layer4', 'layer0', 'init']
LEARNING_RATE_BACKBONE_MULTIPLIER = 0.1  # Backbone 学习率缩放系数

# ==============================================================================
# 8. 环境初始化 
# ==============================================================================
# 自动创建输出目录
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)