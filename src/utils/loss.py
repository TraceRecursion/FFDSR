import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import concurrent.futures
import os
import numpy as np
from functools import partial


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # 将标签转换为one-hot编码
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # 创建mask来忽略未标记的像素(255)
        mask = (targets != 255).float()
        targets_one_hot = F.one_hot(targets * (targets != 255).long(), num_classes).permute(0, 3, 1, 2).float()

        # 计算Dice损失
        intersection = torch.sum(probs * targets_one_hot * mask.unsqueeze(1), dim=(0, 2, 3))
        union = torch.sum(probs * mask.unsqueeze(1), dim=(0, 2, 3)) + torch.sum(targets_one_hot * mask.unsqueeze(1),
                                                                                dim=(0, 2, 3))

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_per_class)

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        # 计算CE损失
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        # 计算概率
        pt = torch.exp(-ce_loss)

        # 计算Focal损失
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 忽略指定索引
        mask = targets != self.ignore_index
        focal_loss = focal_loss * mask.float()

        if self.reduction == 'mean':
            return focal_loss.sum() / mask.sum() if mask.sum() > 0 else focal_loss.sum() * 0.0
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MixedLoss(nn.Module):
    def __init__(self,
                 alpha=0.5,
                 gamma=2.0,
                 class_weights=None,
                 ignore_index=255,
                 smooth=1.0,
                 focal_weight=0.5,
                 dice_weight=0.5):
        super(MixedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma, alpha=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

    def update_weights(self, class_weights):
        """更新损失函数的类别权重"""
        self.focal.alpha = class_weights.to(self.focal.alpha.device)


def _process_batch(batch_indices, dataset, num_classes):
    """处理一批数据样本并计算类别统计"""
    batch_counts = torch.zeros(num_classes)
    for idx in batch_indices:
        _, label, _ = dataset[idx]
        mask = (label != 255)  # 忽略无效像素
        valid_labels = label[mask]

        # 统计每个类别的像素数量
        if len(valid_labels) > 0:
            for c in range(num_classes):
                batch_counts[c] += torch.sum(valid_labels == c).item()
    return batch_counts


def get_class_weights(dataset, num_classes=19, use_predefined=False,
                      min_weight=0.05, max_weight=5.0,
                      use_log_scale=False, mix_ratio=0.0):
    """
    计算Cityscapes数据集的类别权重，使用多线程加速计算

    参数:
        dataset: 数据集对象，用于计算类别分布
        num_classes: 类别数量
        use_predefined: 是否使用预定义权重而不是动态计算
        min_weight: 权重的最小值
        max_weight: 权重的最大值
        use_log_scale: 是否使用对数缩放来压缩权重范围
        mix_ratio: 混合预定义权重的比例 (0.0表示完全使用动态权重，1.0表示完全使用预定义权重)

    返回:
        torch.Tensor: 类别权重向量
    """
    # 预定义的Cityscapes类别权重 (基于类别像素频率的倒数)
    predefined_weights = torch.tensor([
        0.8373, 1.0333, 0.6923, 1.1429, 1.0455, 1.6438,
        1.8462, 1.4545, 0.8000, 1.0323, 0.6897, 1.6667,
        1.9231, 0.8571, 1.8182, 2.0000, 2.1818, 1.8750, 2.0769
    ])

    # 如果没有提供数据集或选择使用预定义权重，则返回预定义值
    if dataset is None or use_predefined:
        print("使用预定义的类别权重")
        return predefined_weights

    # 从数据集动态计算权重
    print("开始计算动态类别权重...")
    print(f"使用完整数据集计算权重，共 {len(dataset)} 个样本")

    # 自动确定并发线程数量
    max_workers = os.cpu_count()
    print(f"使用多线程加速计算，自动检测CPU核心数: {max_workers}")

    # 将数据集样本索引划分为多个批次
    dataset_size = len(dataset)
    batch_size = max(1, dataset_size // (max_workers * 4))  # 每个工作线程处理多个批次
    batches = []
    for i in range(0, dataset_size, batch_size):
        batches.append(list(range(i, min(i + batch_size, dataset_size))))

    class_count = torch.zeros(num_classes)

    # 多线程处理批次
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建处理函数，固定数据集和类别数参数
        process_func = partial(_process_batch, dataset=dataset, num_classes=num_classes)

        # 使用tqdm跟踪多线程进度
        futures = {executor.submit(process_func, batch): batch for batch in batches}

        # 收集结果
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(batches),
                           desc="计算类别分布"):
            batch_counts = future.result()
            class_count += batch_counts

    # 打印各类别像素数量分布
    print("类别像素分布:")
    for c in range(num_classes):
        print(f"类别 {c}: {class_count[c]:.0f} 像素")

    # 避免除零错误
    class_count[class_count == 0] = 1

    # 计算权重 (使用中位数频率平衡法)
    median_freq = torch.median(class_count)
    class_weights = median_freq / class_count

    # 标准化权重，使均值为1
    class_weights = class_weights * (num_classes / torch.sum(class_weights))

    # 如果使用对数缩放
    if use_log_scale:
        print("应用对数缩放来压缩权重范围")
        log_weights = torch.log1p(class_weights)  # log(1+x)避免负值
        class_weights = log_weights * (num_classes / torch.sum(log_weights))
        print(f"对数缩放后的权重: {class_weights}")

    # 混合预定义与动态权重
    if mix_ratio > 0.0:
        print(f"混合预定义权重，混合比例: {mix_ratio:.2f}")
        mixed_weights = mix_ratio * predefined_weights + (1.0 - mix_ratio) * class_weights
        class_weights = mixed_weights
        print(f"混合后的权重: {class_weights}")

    # 限制权重范围，避免极端不平衡
    orig_weights = class_weights.clone()
    class_weights = torch.clamp(class_weights, min=min_weight, max=max_weight)

    # 再次标准化确保均值为1
    class_weights = class_weights * (num_classes / torch.sum(class_weights))

    print(f"原始动态权重: {orig_weights}")
    print(f"最终类别权重: {class_weights}")
    return class_weights


def create_progressive_weight_scheduler(initial_weights, target_weights, total_epochs, warmup_epochs=None):
    """
    创建一个渐进式权重调度器，用于在训练过程中逐渐调整权重

    参数:
        initial_weights: 初始权重
        target_weights: 目标权重
        total_epochs: 总训练轮数
        warmup_epochs: 预热轮数，在此期间线性增加权重影响

    返回:
        function: 一个函数，接收当前epoch作为参数，返回当前应使用的权重
    """
    if warmup_epochs is None:
        warmup_epochs = total_epochs // 2

    def weight_scheduler(epoch):
        if epoch < warmup_epochs:
            # 线性增加目标权重的影响
            alpha = epoch / warmup_epochs
        else:
            # 完全使用目标权重
            alpha = 1.0

        # 计算当前权重
        current_weights = (1.0 - alpha) * initial_weights + alpha * target_weights

        # 确保均值为1
        num_classes = len(current_weights)
        current_weights = current_weights * (num_classes / torch.sum(current_weights))

        return current_weights

    return weight_scheduler
