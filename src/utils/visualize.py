import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt

CATEGORIES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
              'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
              'truck', 'bus', 'train', 'motorcycle', 'bicycle']
COLORS = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
          (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
          (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
          (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]


def visualize_segmentation(pred, orig_img, debug=False):
    """
    将语义分割预测结果可视化到原始图像上 - 优化版本，减少内存使用

    Args:
        pred: 预测的分割图，形状为 (H, W)，值为类别索引
        orig_img: 原始图像，PIL Image对象
        debug: 是否启用调试模式，会显示中间结果

    Returns:
        PIL Image，带有叠加语义分割结果的图像
    """
    try:
        # 转换为numpy数组并确保类型正确
        orig_np = np.array(orig_img, dtype=np.uint8)

        # 调整预测图大小
        if pred.shape[:2] != orig_np.shape[:2]:
            pred_resized = cv2.resize(pred.astype(np.uint8), (orig_np.shape[1], orig_np.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        else:
            pred_resized = pred

        # 创建彩色分割图，避免使用布尔掩码索引
        overlay = np.zeros_like(orig_np, dtype=np.uint8)
        
        # 创建颜色映射表
        color_map = np.zeros((256, 3), dtype=np.uint8)
        for class_id, color in enumerate(COLORS):
            if class_id < len(COLORS):
                color_map[class_id] = color

        # 更安全的颜色索引方法
        valid_indices = (pred_resized >= 0) & (pred_resized < 256)
        if not np.all(valid_indices):
            # 如果有无效索引，先处理它们
            pred_resized = np.clip(pred_resized, 0, 255)

        # 应用颜色映射 - 减少内存使用的方式
        for y in range(pred_resized.shape[0]):
            for x in range(pred_resized.shape[1]):
                class_id = pred_resized[y, x]
                if class_id < len(COLORS):  # 安全检查
                    overlay[y, x] = color_map[class_id]

        # 简化标签部分 - 完全移除或减少标签数量
        # 这里我们选择只显示最大连通区域的标签，减少内存使用
        result_np = orig_np.copy()
        alpha = 0.5
        result_np = (1 - alpha) * orig_np + alpha * overlay
        
        # 限制每个图像最多只添加5个标签
        unique_classes = np.unique(pred_resized)
        unique_classes = unique_classes[unique_classes < len(CATEGORIES)]
        if len(unique_classes) > 5:
            # 如果类别太多，只保留区域最大的5个
            class_areas = {}
            for cls in unique_classes:
                class_areas[cls] = np.sum(pred_resized == cls)
            unique_classes = sorted(class_areas.keys(), key=lambda x: class_areas[x], reverse=True)[:5]
        
        # 创建结果图像并添加文本
        result_img = Image.fromarray(result_np.astype(np.uint8))

        return result_img
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        # 在出错的情况下，返回原图
        return orig_img