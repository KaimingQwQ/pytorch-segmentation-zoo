import numpy as np
import torch

class IOUMetric:
    """
    混淆矩阵法计算 mIoU
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # 初始化混淆矩阵: N x N
        # 行代表 Ground Truth (真实标签)
        # 列代表 Prediction (预测标签)
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # 展平 + 掩码过滤
        # 过滤掉 255 (ignore_index) 以及其他异常值
        mask = (label_true >= 0) & (label_true < self.num_classes)
        
        # 将二维坐标 (true, pred) 映射为一维索引: hist_idx = true * N + pred
        # 然后用 bincount 统计每个索引出现的次数，直接填入混淆矩阵
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        """
        处理一个 Batch 的数据并累积到混淆矩阵
        predictions: [B, H, W] (已经是 argmax 后的索引)
        gts:         [B, H, W]
        """
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        """
        计算各项指标
        """
        # 1. 计算 IoU Per Class
        # IoU = TP / (TP + FP + FN)
        # TP: 对角线元素 diag(hist)
        # FP + FN + TP: 行求和 + 列求和 - 对角线
        
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + 1e-10)

        # 2. 计算 mIoU
        miou = np.nanmean(iu)
        
        # 3. 计算 Accuracy (Pixel Accuracy)
        # acc = diag / sum
        acc = np.diag(self.hist).sum() / (self.hist.sum() + 1e-10)
        return acc, miou, iu