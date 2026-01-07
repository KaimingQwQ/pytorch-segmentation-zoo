import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime

import config
from src.dataset import VOCAugDataset
from src.utils import IOUMetric
from src.logger import ExperimentLogger   
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# 设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import cv2

def get_transforms():
  
    train_transform = A.Compose([
        # 1. 几何变换
        A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
        A.PadIfNeeded(min_height=config.img_size, min_width=config.img_size, 
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255),
        A.RandomCrop(height=config.img_size, width=config.img_size),
        A.HorizontalFlip(p=0.5),

        # 2. 色彩增强
        A.OneOf([
            # 调整亮度、对比度、饱和度、色相
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
            # 随机调整 Gamma 值，模拟不同光照环境
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
    """验证循环"""
    model.eval()
    val_loss = 0
    iou_metric = IOUMetric(num_classes)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation', unit='batch', leave=False):
            images, true_masks = batch
            images = images.to(device, dtype=torch.float32)
            true_masks = true_masks.to(device, dtype=torch.long)

            # 验证阶段也可以开 autocast 省一点显存，但为了精度通常 FP32 也可以
            # 这里为了保持纯净的评估，保持原样（FP32）
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
    logger.log_text(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    # 4. 优化器、调度器与混合精度 Scaler
    # ---------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS - warmup_epochs, eta_min=1e-6)
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # ---------------------------------------------------
    # 5. 训练循环
    # ---------------------------------------------------
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        
    best_miou = 0.0
    logger.log_text("\n Start Training  (开启混合精度训练)...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}', unit='batch') as pbar:
            for batch in train_loader:
                images, true_masks = batch
                images = images.to(device, dtype=torch.float32)
                true_masks = true_masks.to(device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    masks_pred = model(images)
                    
                    if isinstance(masks_pred, dict):
                        loss = criterion(masks_pred['out'], true_masks) + 0.5 * criterion(masks_pred['aux'], true_masks)
                    else:
                        loss = criterion(masks_pred, true_masks)

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
        # 6. 验证与保存
        # ---------------------------------------------------
        val_loss, val_acc, val_miou, val_class_iou = evaluate(model, val_loader, device, criterion, config.NUM_CLASSES)
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
            
            print(f"  Best Model Saved: {save_path} (mIoU: {best_miou:.4f})")

        logger.log_csv(epoch+1, train_loss_avg, val_loss, val_acc, val_miou, best_miou, current_lr, val_class_iou)
        
        log_msg = (f"[Epoch {epoch+1}] Train Loss: {train_loss_avg:.4f} | "
                   f"Val mIoU: {val_miou:.4f} | LR: {current_lr:.6f} {'🔥 Best' if is_best else ''}")
        logger.log_text(log_msg)

    logger.log_text(f"\n Training Finished. Best mIoU: {best_miou:.4f}")
    print(f" Logs saved to: {logger.output_dir}")

if __name__ == '__main__':
    train_model()