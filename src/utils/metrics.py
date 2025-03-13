import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def calculate_miou(pred, target, num_classes=19):
    pred = pred.argmax(dim=1).flatten()
    target = target.flatten()
    mask = (target != 255)
    pred, target = pred[mask], target[mask]
    if len(pred) == 0:
        return 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for i in range(num_classes):
        mask = (pred == i)
        confusion_matrix[i] = torch.bincount(target[mask], minlength=num_classes)[:num_classes]
    hist = confusion_matrix.numpy()
    intersection = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    iou = intersection / (union + 1e-10)
    return np.nanmean(iou)


def calculate_class_iou(preds, targets, num_classes=19):
    """
    计算每个类别的IoU和整体mIoU

    参数:
        preds: 预测结果 tensor [N, C, H, W]
        targets: 标签 tensor [N, H, W]
        num_classes: 类别数量

    返回:
        class_ious: 每个类别的IoU
        miou: 平均IoU
        confusion_matrix: 混淆矩阵
    """
    if preds.dim() == 4 and preds.shape[1] > 1:  # [N, C, H, W] 格式
        preds = preds.argmax(dim=1)  # [N, H, W]

    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    preds = preds.flatten()
    targets = targets.flatten()
    valid_mask = targets != 255
    preds = preds[valid_mask]
    targets = targets[valid_mask]

    for t, p in zip(targets.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
    class_ious = intersection.float() / (union.float() + 1e-10)
    miou = torch.nanmean(class_ious)

    return class_ious, miou, confusion_matrix


def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
    """
    绘制混淆矩阵图

    参数:
        confusion_matrix: 混淆矩阵 tensor
        class_names: 类别名称列表
        save_path: 保存路径
    """
    plt.figure(figsize=(14, 12))
    confusion_matrix = confusion_matrix.cpu().numpy()

    # 归一化
    row_sums = confusion_matrix.sum(axis=1)
    norm_conf_mx = confusion_matrix / (row_sums[:, np.newaxis] + 1e-7)

    sns.heatmap(norm_conf_mx, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')  # 中文标签
    plt.ylabel('真实标签')  # 中文标签
    plt.title('语义分割混淆矩阵 (行归一化)')  # 中文标题
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"混淆矩阵已保存至 {save_path}")
    else:
        plt.show()
    plt.close()