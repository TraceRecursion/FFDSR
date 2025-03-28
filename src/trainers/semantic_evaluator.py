import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager  # 添加字体管理
from datetime import datetime

from ..datasets.semantic_dataset import CityscapesDataset
from ..models.semantic_model import get_semantic_model
from ..utils.metrics import plot_confusion_matrix
from ..utils.common import get_device
from ..utils.visualize import visualize_segmentation
from PIL import Image

# 设置支持中文的字体（Windows常用字体：SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class SemanticEvaluator:
    def __init__(self, config):
        self.config = config
        self.device, device_name = get_device()
        print(f"使用设备: {device_name}")

        # 获取预训练权重路径
        if 'model_path' in self.config['model']:
            checkpoint_path = self.config['model']['model_path']
        elif 'pretrained_weights' in self.config['model']:
            checkpoint_path = self.config['model']['pretrained_weights']
        else:
            print("错误: 配置文件中缺少模型权重路径，请检查配置文件")
            checkpoint_path = None

        # 初始化模型
        self.model = get_semantic_model(
            self.config['model']['num_classes'],
            checkpoint_path
        ).to(self.device)

        print("模型初始化完成")
        self.model.eval()

        # Cityscapes类别名称
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]

    def evaluate(self):
        # 创建验证数据集
        dataset = CityscapesDataset(
            self.config['data']['val_img_dir'],
            self.config['data']['val_label_dir'],
            crop_size=None,  # 不进行裁剪，使用全尺寸图像
            test_mode=True
        )

        print(f"评估数据集大小: {len(dataset)}张图像")

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 创建结果目录
        os.makedirs(self.config['output']['dir'], exist_ok=True)

        # 初始化混淆矩阵
        num_classes = self.config['model']['num_classes']
        confusion_matrix = torch.zeros(num_classes, num_classes,
                                       dtype=torch.long, device='cpu')

        progress_bar = tqdm(dataloader, desc="语义分割评估")

        with torch.no_grad():
            for batch_idx, (imgs, labels, img_names) in enumerate(progress_bar):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # 模型预测
                outputs = self.model(imgs)['out']
                preds = outputs.argmax(dim=1)  # [B, H, W]

                # 增量更新混淆矩阵
                preds = preds.cpu()
                labels = labels.cpu()
                valid_mask = labels != 255
                preds = preds[valid_mask]
                labels = labels[valid_mask]

                # 使用torch.bincount优化混淆矩阵计算
                idx = labels * num_classes + preds
                counts = torch.bincount(idx, minlength=num_classes * num_classes)
                confusion_matrix += counts.view(num_classes, num_classes)

                # 可选：保存分割结果图像
                if self.config['output'].get('save_results', False):
                    save_dir = os.path.join(self.config['output']['dir'], 'predictions')
                    os.makedirs(save_dir, exist_ok=True)

                    for i in range(len(img_names)):
                        pred = preds[i].numpy()
                        orig_img = Image.open(os.path.join(self.config['data']['val_img_dir'], img_names[i]))
                        result_img = visualize_segmentation(pred, orig_img)
                        result_img.save(os.path.join(save_dir, f"pred_{img_names[i]}"), quality=90)

        # 计算IoU
        print("\n计算每个类别的IoU和mIoU...")
        class_ious, miou = self._calculate_metrics(confusion_matrix)

        # 打印结果
        print(f"\n整体 mIoU: {miou.item():.4f}")
        print("\n各类别 IoU:")

        # 创建类别IoU结果表格
        results_data = []
        for i, class_name in enumerate(self.class_names):
            iou_val = class_ious[i].item()
            print(f"{class_name}: {iou_val:.4f}")
            results_data.append({
                'Class ID': i,
                'Class Name': class_name,
                'IoU': iou_val
            })

        # 保存结果为CSV
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('IoU', ascending=False)  # 按IoU值降序排序

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.config['output']['dir'], f"iou_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n详细结果已保存至: {results_path}")

        # 可选：绘制混淆矩阵
        if self.config['output'].get('save_confusion_matrix', False):
            confusion_matrix_path = os.path.join(self.config['output']['dir'], f"confusion_matrix_{timestamp}.png")
            plot_confusion_matrix(confusion_matrix, self.class_names, confusion_matrix_path)

        # 绘制IoU条形图
        plt.figure(figsize=(14, 8))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.class_names)))
        bars = plt.bar(results_df['Class Name'], results_df['IoU'], color=colors)
        plt.xlabel('类别')
        plt.ylabel('IoU')
        plt.title('各类别IoU值')  # 中文标题
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=miou.item(), color='r', linestyle='-', label=f'mIoU: {miou.item():.4f}')
        plt.legend()
        plt.tight_layout()

        chart_path = os.path.join(self.config['output']['dir'], f"class_iou_chart_{timestamp}.png")
        plt.savefig(chart_path, dpi=150)
        print(f"IoU条形图已保存至: {chart_path}")

        return miou.item(), class_ious.tolist()

    def _calculate_metrics(self, confusion_matrix):
        """从混淆矩阵计算IoU"""
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(0) + confusion_matrix.sum(1) - intersection
        class_ious = intersection.float() / (union.float() + 1e-10)
        miou = torch.nanmean(class_ious)
        return class_ious, miou