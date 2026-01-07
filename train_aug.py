import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime
import cv2

import config

from src.dataset import VOCAugDataset
from src.utils import IOUMetric
from src.logger import ExperimentLogger   
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from src.losses import CE_DiceLoss, CE_LovaszLoss

# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def get_transforms():
    """
    定义数据增强策略
    """
    train_transform = A.Compose([
        # 1. 几何变换
        A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
        A.PadIfNeeded(min_height=config.img_size, min_width=config.img_size, 
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
        A.RandomCrop(height=config.img_size, width=config.img_size),
        A.HorizontalFlip(p=0.5),
        # 2. 色彩增强
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            A.RandomGamma(p=1),
        ], p=0.5),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return train_transform, val_transform

def evaluate(model, loader, device, criterion, num_classes):
    model.eval()
    val_loss = 0
    iou_metric = IOUMetric(num_classes)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation', unit='batch', leave=False):
            images, true_masks = batch
            images = images.to(device, dtype=torch.float32)
            true_masks = true_masks.to(device, dtype=torch.long)

            masks_pred = model(images)

            if isinstance(masks_pred, dict):
                masks_pred = masks_pred['out']

            loss = criterion(masks_pred, true_masks)
            val_loss += loss.item()
            
            pred_indices = torch.argmax(masks_pred, dim=1)
            iou_metric.add_batch(pred_indices.cpu().numpy(), true_masks.cpu().numpy())
            
    acc, miou, iou_per_class = iou_metric.evaluate()
    return val_loss / len(loader), acc, miou, iou_per_class

def train_model():
    # ---------------------------------------------------
    # 1. 硬件与模型初始化
    # ---------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = config.MODEL 
    except AttributeError:
        model = config.MODLE 
        
    model_name = model.__class__.__name__

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) 
    
    model.to(device)

    # ---------------------------------------------------
    # 2. 初始化日志
    # ---------------------------------------------------
    logger = ExperimentLogger(model_name)
    logger.log_text("=" * 60)
    logger.log_text(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 读取策略配置 
    use_lovasz = getattr(config, 'USE_LOVASZ_FINETUNE', False)
    switch_epoch = getattr(config, 'LOVASZ_SWITCH_EPOCH', int(config.NUM_EPOCHS * 0.5))
    lovasz_weight = getattr(config, 'LOVASZ_WEIGHT', 0.75)
    
    logger.log_text(f" Base Loss: {config.LOSS_TYPE}")
    logger.log_text(f" Strategy: {'Lovász Finetuning ON' if use_lovasz else 'Standard Training'}")
    if use_lovasz:
        logger.log_text(f"   -> Switch Epoch: {switch_epoch} | Weight: {lovasz_weight}")
    logger.log_text("=" * 60)

    # ---------------------------------------------------
    # 3. 数据加载
    # ---------------------------------------------------
    train_transform, val_transform = get_transforms()
    
    train_dataset = VOCAugDataset(config.DATA_DIR, split='train_aug', transform=train_transform)
    val_dataset = VOCAugDataset(config.DATA_DIR, split='val', transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=True, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=True
    )
    
    logger.log_text(f" Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # ---------------------------------------------------
    # 4. 优化器与参数分组
    # ---------------------------------------------------
    if isinstance(model, nn.DataParallel):
        _model = model.module
    else:
        _model = model
    #指定要减速的参数
    backbone_keywords = config.backbone_keywords
    backbone_params = []
    head_params = []

    for name, param in _model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = any(k in name for k in backbone_keywords)
        if is_backbone:
            backbone_params.append(param)
        else:
            head_params.append(param)

    print(f" Optim Params: Backbone has {len(backbone_params)} tensors, Head has {len(head_params)} tensors.")
    
    params_list = [
        {'params': backbone_params, 'lr': config.LEARNING_RATE * config.LEARNING_RATE_BACKBONE_MULTIPLIER}, 
        {'params': head_params, 'lr': config.LEARNING_RATE}
    ]

    optimizer = optim.AdamW(params_list, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    warmup_epochs = config.WARMUP_EPOCHS if hasattr(config, 'WARMUP_EPOCHS') else 10
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )
    
    # ---------------------------------------------------
    # 5. 损失函数策略定义
    # ---------------------------------------------------  
    # A. 基础阶段 Loss (Stable)
    if config.LOSS_TYPE == 'ce_dice':
        print(f" [Phase 1] Stable Loss: CE + {config.DICE_WEIGHT}*Dice")
        logger.log_text("\n Using Dice Loss for Phase 1!")
        criterion_stable = CE_DiceLoss(num_classes=config.NUM_CLASSES, ignore_index=255, dice_weight=config.DICE_WEIGHT)
    else:
        logger.log_text(" Using Standard CE Loss for Phase 1!")
        print(" [Phase 1] Stable Loss: Standard CE")
        criterion_stable = nn.CrossEntropyLoss(ignore_index=255)

    # B. 微调阶段 Loss (Finetune)
    if use_lovasz:
        print(f" [Phase 2] Finetune Loss: CE + {lovasz_weight}*Lovász (Start @ Epoch {switch_epoch})")
        logger.log_text("\n Using Lovász Loss for Finetuning Phase!")
        criterion_finetune = CE_LovaszLoss(ignore_index=255, lovasz_weight=lovasz_weight)
    else:
        criterion_finetune = criterion_stable # 如果不启用，Phase 2 和 Phase 1 一样

    # C. 辅助 Loss 
    criterion_aux = nn.CrossEntropyLoss(ignore_index=255)

    # ---------------------------------------------------
    # 6. 训练循环
    # ---------------------------------------------------
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    best_miou = 0.0
    logger.log_text("\n Start Training Loop (开启混合精度)...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        # -----------------------------------------------
        # 策略切换逻辑
        # -----------------------------------------------
        if use_lovasz and epoch >= switch_epoch:
            current_criterion = criterion_finetune
            strategy_name = "Finetune(Lovász)"
        else:
            current_criterion = criterion_stable
            strategy_name = "Stable(Base)"
            
        # -----------------------------------------------

        # 获取学习率显示
        if len(optimizer.param_groups) > 1:
            current_lr = optimizer.param_groups[1]['lr'] 
            backbone_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
            backbone_lr = current_lr
        desc_str = f'Epoch {epoch+1}/{config.NUM_EPOCHS} [{strategy_name}]'

        with tqdm(total=len(train_loader), desc=desc_str, unit='batch') as pbar:
            for batch in train_loader:
                images, true_masks = batch
                images = images.to(device, dtype=torch.float32)
                true_masks = true_masks.to(device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)
                # 混合精度前向
                with torch.cuda.amp.autocast():
                    masks_pred = model(images)
                    
                    if isinstance(masks_pred, dict):
                        main_pred = masks_pred['out']
                        aux_pred = masks_pred['aux']
                        loss_main = current_criterion(main_pred, true_masks)
                        loss_aux = criterion_aux(aux_pred, true_masks)
                        loss = loss_main + 0.5 * loss_aux
                    else:
                        loss = current_criterion(masks_pred, true_masks)
        
                # 混合精度反向
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item(), lr=current_lr)
                pbar.update(1)

        scheduler.step()

        # ---------------------------------------------------
        # 7. 验证与保存
        # ---------------------------------------------------
        val_loss, val_acc, val_miou, val_class_iou = evaluate(model, val_loader, device, current_criterion, config.NUM_CLASSES)
        train_loss_avg = epoch_loss / len(train_loader)
        
        is_best = val_miou > best_miou
        
        if is_best:
            best_miou = val_miou
            save_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
            
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            torch.save({
                'epoch': epoch,
                'model_state': state_dict,
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'best_miou': best_miou,
            }, save_path)
            
            print(f"   Best Model Saved: {save_path} (mIoU: {best_miou:.4f})")

        logger.log_csv(epoch+1, train_loss_avg, val_loss, val_acc, val_miou, best_miou, current_lr, val_class_iou)
        
        log_msg = (f"[Epoch {epoch+1}] [{strategy_name}] Loss: {train_loss_avg:.4f} | "
                   f"mIoU: {val_miou:.4f} | "
                   f"HeadLR: {current_lr:.1e} | BackLR: {backbone_lr:.1e} "
                   f"{'🔥 Best' if is_best else ''}")
        logger.log_text(log_msg)

    logger.log_text(f"\n Training Finished. Best mIoU: {best_miou:.4f}")
    print(f" Logs saved to: {logger.output_dir}")

if __name__ == '__main__':
    train_model()