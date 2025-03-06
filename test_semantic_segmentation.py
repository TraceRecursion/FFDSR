import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import datetime

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 数据集相对路径
base_data_dir = os.path.join(current_dir, '../../Documents/数据集/CitySpaces')
test_img_dir = os.path.join(base_data_dir, 'leftImg8bit_trainvaltest/leftImg8bit/test')
test_label_dir = os.path.join(base_data_dir, 'gtFine_trainvaltest/gtFine/test')

# 输出目录
output_dir = os.path.join(current_dir, 'outputs/semantic')
os.makedirs(output_dir, exist_ok=True)

# Cityscapes 类别和颜色映射
categories = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
colors = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

# Cityscapes 数据集类
class CityscapesTestDataset(Dataset):
    def __init__(self, img_dir, label_dir, crop_size=512):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.crop_size = crop_size
        self.images = []
        self.labels = []

        print(f"初始化测试数据集: {img_dir}")
        for city in tqdm(os.listdir(img_dir), desc="扫描城市"):
            city_img_dir = os.path.join(img_dir, city)
            city_label_dir = os.path.join(label_dir, city)
            if os.path.isdir(city_img_dir):
                for img_file in os.listdir(city_img_dir):
                    if img_file.endswith('leftImg8bit.png'):
                        label_file = img_file.replace('leftImg8bit.png', 'gtFine_labelIds.png')
                        self.images.append(os.path.join(city_img_dir, img_file))
                        self.labels.append(os.path.join(city_label_dir, label_file))

        print(f"测试数据集加载完成: {len(self.images)} 张图像")
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # 转换为张量
        img_tensor = self.transform_img(img)
        label_np = np.array(label, dtype=np.uint8)

        return img_tensor, label_np, img, os.path.basename(img_path)

# 自定义 collate_fn
def custom_collate_fn(batch):
    img_tensors, label_nps, orig_imgs, img_names = zip(*batch)
    return (torch.stack(img_tensors),  # 张量堆叠
            torch.from_numpy(np.stack(label_nps)),  # NumPy 数组堆叠
            list(orig_imgs),  # 保持 PIL 图像为列表
            list(img_names))  # 保持文件名为列表

# 测试函数
def test_semantic_segmentation():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用设备: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")

    # 加载模型，与训练时一致
    model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1').to(device)
    model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)  # 修改为主分类器
    model.aux_classifier[4] = nn.Conv2d(256, 19, kernel_size=1)  # 修改为辅助分类器
    model = model.to(device)

    model_path = os.path.join(current_dir, "models/best_semantic_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"最佳模型未找到: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"加载模型: {model_path}")

    # 数据加载
    batch_size = 1  # 单张测试，避免显存溢出
    test_dataset = CityscapesTestDataset(test_img_dir, test_label_dir, crop_size=512)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                             collate_fn=custom_collate_fn)

    # 测试循环
    with torch.no_grad():
        for img_tensor, label_np, orig_img, img_name in tqdm(test_loader, desc="测试中"):
            img_tensor = img_tensor.to(device)
            outputs = model(img_tensor)['out']
            pred = outputs.argmax(dim=1).cpu().numpy()[0]  # [H, W]

            # 可视化结果
            result_img = visualize_segmentation(pred, orig_img[0])
            output_path = os.path.join(output_dir, f"pred_{img_name[0]}")
            result_img.save(output_path)
            print(f"保存结果: {output_path}")

def visualize_segmentation(pred, orig_img):
    # 将原始图像转换为 NumPy 数组
    orig_np = np.array(orig_img)
    result_np = orig_np.copy()

    # 创建半透明掩码层
    overlay = np.zeros_like(orig_np, dtype=np.uint8)

    # 为每个类别绘制像素级色块
    for class_id, (category, color) in enumerate(zip(categories, colors)):
        mask = (pred == class_id)
        if np.any(mask):
            # 应用像素级掩码
            overlay[mask] = color  # 直接设置颜色

            # 计算掩码中心用于文字
            y, x = np.where(mask)
            if len(x) > 0 and len(y) > 0:
                center_x = int(x.mean())
                center_y = int(y.mean())

                # 转换为 PIL 图像以绘制文字
                result_img = Image.fromarray(result_np)
                draw = ImageDraw.Draw(result_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)  # 尝试加载字体
                except:
                    font = ImageFont.load_default()  # 默认字体

                text_bbox = draw.textbbox((center_x, center_y), category, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle([center_x - text_width // 2, center_y - text_height // 2,
                                center_x + text_width // 2, center_y + text_height // 2],
                               fill=color + (200,))  # 文字背景
                draw.text((center_x - text_width // 2, center_y - text_height // 2), category, font=font, fill=(255, 255, 255, 255))
                result_np = np.array(result_img)

    # 合并原始图像和掩码层（半透明效果）
    alpha = 0.5  # 透明度
    result_np = (1 - alpha) * orig_np + alpha * overlay
    result_np = result_np.astype(np.uint8)

    return Image.fromarray(result_np)

if __name__ == "__main__":
    test_semantic_segmentation()