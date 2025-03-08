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
    将语义分割预测结果可视化到原始图像上

    Args:
        pred: 预测的分割图，形状为 (H, W)，值为类别索引
        orig_img: 原始图像，PIL Image对象
        debug: 是否启用调试模式，会显示中间结果

    Returns:
        PIL Image，带有叠加语义分割结果的图像
    """
    orig_np = np.array(orig_img)

    if debug:
        print(f"原始图像尺寸: {orig_np.shape}")
        print(f"预测分割图尺寸: {pred.shape}")

    if pred.shape[:2] != orig_np.shape[:2]:
        print(f"调整预测图尺寸: {pred.shape} -> {orig_np.shape[:2]}")
        pred_resized = cv2.resize(pred, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        pred_resized = pred

    overlay = np.zeros_like(orig_np, dtype=np.uint8)
    result_np = orig_np.copy()

    for class_id, (category, color) in enumerate(zip(CATEGORIES, COLORS)):
        mask = (pred_resized == class_id)
        if np.any(mask):
            overlay[mask] = color

            y, x = np.where(mask)
            if len(x) > 0:
                center_x, center_y = int(x.mean()), int(y.mean())
                result_img = Image.fromarray(result_np)
                draw = ImageDraw.Draw(result_img)

                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except IOError:
                    font = ImageFont.load_default()

                text_bbox = draw.textbbox((center_x, center_y), category, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                draw.rectangle([center_x - text_width // 2, center_y - text_height // 2,
                                center_x + text_width // 2, center_y + text_height // 2],
                               fill=color + (200,))
                draw.text((center_x - text_width // 2, center_y - text_height // 2),
                          category, font=font, fill=(255, 255, 255))
                result_np = np.array(result_img)

    if debug:
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(orig_np)
        plt.title("原始图像")
        plt.subplot(1, 3, 2)
        plt.imshow(overlay)
        plt.title("分割结果")
        plt.subplot(1, 3, 3)
        plt.imshow((0.5 * orig_np + 0.5 * overlay).astype(np.uint8))
        plt.title("叠加结果")
        plt.tight_layout()
        plt.show()

    alpha = 0.5
    result_np = (1 - alpha) * orig_np + alpha * overlay

    return Image.fromarray(result_np.astype(np.uint8))