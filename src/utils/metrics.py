import torch
import numpy as np

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