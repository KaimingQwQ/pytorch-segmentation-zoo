import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import re
import sys

# 设置字体以支持中文显示 (可选，视系统而定)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def plot_from_csv(csv_path):
    """
    从给定的 CSV 文件中读取训练日志数据，并生成训练损失、验证准确率、验证 mIoU 以及各类别 IoU 的曲线图。

    Args:
        csv_path (str): CSV 文件的路径。
                        文件应包含 'Epoch', 'Train_Loss', 'Val_Loss', 'Val_Acc', 'Val_mIoU' 等列。
                        如果包含 'IoU_' 开头的列，也会生成每类 IoU 的柱状图。
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: 找不到文件 {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取日志: {csv_path}")
        print(f"   包含列: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error: 读取 CSV 失败: {e}")
        return
    filename = os.path.basename(csv_path)
    name_no_ext = os.path.splitext(filename)[0]
    subdir_name = re.sub(r'_[\d_]+$', '', name_no_ext)
    
    if not subdir_name:
        subdir_name = name_no_ext
        
    base_dir = os.path.dirname(csv_path)
    save_dir = os.path.join(base_dir, 'plots', subdir_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📂 创建绘图目录: {save_dir}")

    try:
        epochs = df['Epoch']
        plt.style.use('bmh') 
    except KeyError:
        print("❌ Error: CSV 中缺少 'Epoch' 列，无法绘图。")
        return

    # --- 图 1: Loss 曲线 ---
    if 'Train_Loss' in df.columns and 'Val_Loss' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df['Train_Loss'], label='Train Loss', color='#d62728', linewidth=2)
        plt.plot(epochs, df['Val_Loss'], label='Val Loss', color='#1f77b4', linewidth=2, linestyle='--')
        plt.title('Training and Validation Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, '01_loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # --- 图 2: Accuracy 曲线 ---
    if 'Val_Acc' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df['Val_Acc'], label='Pixel Accuracy', color='#2ca02c', linewidth=2)
        plt.title('Validation Pixel Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, '02_accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # --- 图 3: mIoU 曲线 ---
    if 'Val_mIoU' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df['Val_mIoU'], label='Val mIoU', color='#9467bd', linewidth=2)
        
        best_epoch_idx = df['Val_mIoU'].idxmax()
        best_miou = df['Val_mIoU'].iloc[best_epoch_idx]
        best_epoch = df['Epoch'].iloc[best_epoch_idx]
        
        plt.scatter(best_epoch, best_miou, color='red', s=100, zorder=5, label=f'Best: {best_miou:.4f} @ Ep {best_epoch}')
        
        plt.title('Validation Mean IoU (mIoU)', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('mIoU', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, '03_miou_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # --- 图 4: 最佳 Epoch 的各类别 IoU  ---
        best_row = df.iloc[best_epoch_idx]
        iou_cols = [col for col in df.columns if col.startswith('IoU_')]
        
        if iou_cols:
            class_names = [col.replace('IoU_', '') for col in iou_cols]
            iou_values = best_row[iou_cols].values.astype(float)
            
            sorted_indices = np.argsort(iou_values)
            sorted_names = [class_names[i] for i in sorted_indices]
            sorted_values = [iou_values[i] for i in sorted_indices]
            
            plt.figure(figsize=(10, max(6, len(sorted_names) * 0.4))) 
            bars = plt.barh(sorted_names, sorted_values, color='#17becf', alpha=0.8)
            
            plt.title(f'Per-Class IoU (at Best Epoch {best_epoch})', fontsize=14)
            plt.xlabel('IoU', fontsize=12)
            plt.xlim(0, 1.05) 
            plt.grid(axis='x', alpha=0.3)
            
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{width:.3f}', 
                         va='center', fontsize=9)
            
            plt.savefig(os.path.join(save_dir, '04_class_iou_best.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print(" CSV 中未发现类别 IoU 数据 (列名需以 'IoU_' 开头)，跳过类别绘图。")

    print(f"\n 所有图片已生成完毕，保存在: {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="绘制训练日志 CSV 的曲线图")
    parser.add_argument('--csv', type=str, default=None, help='CSV 文件的路径')
    args = parser.parse_args()
    
    if args.csv:
        target_path = args.csv
    else:
        # 默认文件路径
        target_path = 'output/RefineNet_ResNet101_Finetune_20260103_135550.csv'
    
    plot_from_csv(target_path)