import os
import csv
import numpy as np
from datetime import datetime
import config  

class ExperimentLogger:
    """
    实验日志系统：同时支持 TXT 文本日志和 CSV 数据记录
    """
    def __init__(self, model_name):
        # 1. 确定输出目录
        self.output_dir = os.path.join(config.ROOT_DIR, 'output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # 2. 生成带时间戳的文件名
        # 格式: 模型名_日期_时间.csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{model_name}_{timestamp}"
        
        self.txt_path = os.path.join(self.output_dir, f"{base_name}.txt")
        self.csv_path = os.path.join(self.output_dir, f"{base_name}.csv")
        
        print(f"📝 Text Log: {self.txt_path}")
        print(f"📊 CSV Data: {self.csv_path}")

        # 3. 初始化 CSV 文件头
        self._init_csv_header()

    def _init_csv_header(self):
        """
        初始化 CSV 表头
        """
        # 通用指标
        headers = ['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Acc', 'Val_mIoU', 'Best_mIoU', 'LR']
        try:
            VOC_CLASSES = [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
                'sofa', 'train', 'tvmonitor'
            ]
            headers.extend([f"IoU_{cls}" for cls in VOC_CLASSES])
        except:
            headers.append("Class_IoUs") # Fallback

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)

    def log_text(self, msg, print_to_console=True):
        """
        记录文字日志到 TXT
        """
        with open(self.txt_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        if print_to_console:
            print(msg)

    def log_csv(self, epoch, train_loss, val_loss, val_acc, val_miou, best_miou, lr, class_ious):
        """
        记录数值数据到 CSV
        """
        row = [
            epoch, 
            f"{train_loss:.6f}", 
            f"{val_loss:.6f}", 
            f"{val_acc:.6f}", 
            f"{val_miou:.6f}", 
            f"{best_miou:.6f}", 
            f"{lr:.8f}"
        ]
        
        # 处理类别 IoU (处理 NaN)
        clean_ious = [f"{iou:.6f}" if not np.isnan(iou) else "0.000000" for iou in class_ious]
        row.extend(clean_ious)

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)