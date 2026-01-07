import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: [B, C, H, W] (Logits)
        # target: [B, H, W] (Long)
        pred = F.softmax(pred, dim=1)
        
        valid_mask = (target != self.ignore_index)
        target = target * valid_mask # 把 ignore 的地方置 0 (防止 one_hot 报错)

        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        valid_mask = valid_mask.unsqueeze(1).float()
        
        pred = pred * valid_mask
        target_onehot = target_onehot * valid_mask

        intersection = (pred * target_onehot).sum(dim=(0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target_onehot.sum(dim=(0, 2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()

class CE_DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, dice_weight=0.4):
        super(CE_DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_dice = self.dice(pred, target)
        
        return loss_ce + self.dice_weight * loss_dice
    
def lovasz_grad(gt_sorted):
    """
    计算 Lovász 扩展的梯度
    gt_sorted: 排序后的真实标签误差
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    多分类 Lovász-Softmax 核心计算
    probas: [N, C] Softmax 后的概率
    labels: [N] 真实标签
    classes: 'all' 对所有类别求平均，'present' 只对当前 batch 出现的类别求平均
    """
    if probas.numel() == 0:
        return probas * 0.
        
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes == 'all' else torch.unique(labels)
    
    for c in class_to_sum:
        fg = (labels == c).float()  
        if (classes == 'present' and fg.sum() == 0):
            continue
            
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        losses.append(loss)
        
    return torch.stack(losses).mean()

def flatten_probas(probas, labels, ignore=None):
    """
    将预测和标签展平，并移除 ignore_index
    """
    if probas.dim() == 3:
        # [B, H, W] -> [B, 1, H, W]
        probas = probas.unsqueeze(1)
    
    B, C, H, W = probas.size()
    # [B, C, H, W] -> [B, H, W, C] -> [N, C]
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    # [B, H, W] -> [N]
    labels = labels.view(-1)
    
    if ignore is None:
        return probas, labels
        
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, ignore_index=255, classes='present'):
        """
        Lovász-Softmax Loss
        Args:
            ignore_index: 忽略的类别索引
            classes: 'all' (所有类别平均) 或 'present' (仅当前Batch存在的类别平均)
                     推荐使用 'present'，训练更稳定，特别是 Batch Size 较小时。
        """
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, pred, target):
        # pred: [B, C, H, W] (Logits)
        # target: [B, H, W] (Long)

        probs = F.softmax(pred, dim=1)

        probs_flat, labels_flat = flatten_probas(probs, target, self.ignore_index)

        loss = lovasz_softmax_flat(probs_flat, labels_flat, classes=self.classes)
        return loss


class CE_LovaszLoss(nn.Module):
    """
    CrossEntropy + Lovász Loss 组合
    建议：先用纯 CE 训练几个 Epoch，然后再换成这个 Loss 进行微调。
    """
    def __init__(self, ignore_index=255, lovasz_weight=0.75):
        super(CE_LovaszLoss, self).__init__()
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight
        
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lovasz = LovaszSoftmaxLoss(ignore_index=ignore_index, classes='present')

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_lovasz = self.lovasz(pred, target)

        # 注意：Lovasz 的数值通常比 CE 小，所以权重有时需要给大一点，或者保持 1:1
        return loss_ce + self.lovasz_weight * loss_lovasz