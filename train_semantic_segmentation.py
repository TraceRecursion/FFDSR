import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import datetime

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 数据集相对路径
base_data_dir = os.path.join(current_dir, '../../Documents/数据集/CitySpaces')
train_img_dir = os.path.join(base_data_dir, 'leftImg8bit_trainvaltest/leftImg8bit/train')
train_label_dir = os.path.join(base_data_dir, 'gtFine_trainvaltest/gtFine/train')
val_img_dir = os.path.join(base_data_dir, 'leftImg8bit_trainvaltest/leftImg8bit/val')
val_label_dir = os.path.join(base_data_dir, 'gtFine_trainvaltest/gtFine/val')

# Cityscapes 标签映射
label_mapping = {
    7: 0,  # road
    8: 1,  # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18  # bicycle
}


# Cityscapes 数据集类（完整数据集，无预加载）
class CityscapesDataset(Dataset):
    def __init__(self, img_dir, label_dir, crop_size=512, batch_size=16):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.images = []
        self.labels = []

        print(f"初始化数据集: {img_dir}")
        for city in tqdm(os.listdir(img_dir), desc="扫描城市"):
            city_img_dir = os.path.join(img_dir, city)
            city_label_dir = os.path.join(label_dir, city)
            if os.path.isdir(city_img_dir):
                for img_file in os.listdir(city_img_dir):
                    if img_file.endswith('leftImg8bit.png'):
                        label_file = img_file.replace('leftImg8bit.png', 'gtFine_labelIds.png')
                        self.images.append(os.path.join(city_img_dir, img_file))
                        self.labels.append(os.path.join(city_label_dir, label_file))

        print(f"数据集加载完成: {len(self.images)} 张图像")
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_item(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])
        img = self.transform_img(img)
        label = np.array(label, dtype=np.uint8)
        mapped_label = np.full(label.shape, 255, dtype=np.uint8)
        for orig_id, train_id in label_mapping.items():
            mapped_label[label == orig_id] = train_id
        label = torch.from_numpy(mapped_label).long()
        return img, label

    def __len__(self):
        return len(self.images)  # 使用完整数据集

    def __getitem__(self, idx):
        img, label = self._load_item(idx)

        h, w = img.shape[1:]
        if h > self.crop_size and w > self.crop_size:
            top = np.random.randint(0, h - self.crop_size)
            left = np.random.randint(0, w - self.crop_size)
            img = img[:, top:top + self.crop_size, left:left + self.crop_size]
            label = label[top:top + self.crop_size, left:left + self.crop_size]

        return img, label


# 计算 mIoU
def calculate_miou(pred, target, num_classes=19):
    pred = pred.argmax(dim=1).flatten()
    target = target.flatten()
    mask = (target != 255)
    pred = pred[mask]
    target = target[mask]
    if len(pred) == 0:
        return 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for i in range(num_classes):
        mask = (pred == i)
        confusion_matrix[i] = torch.bincount(target[mask], minlength=num_classes)[0:num_classes]

    hist = confusion_matrix.numpy()
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou = intersection / (union + 1e-10)
    miou = np.nanmean(iou)
    return miou


# 训练函数
def train_semantic_segmentation():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用设备: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用设备: MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("没有可用的GPU，使用设备: CPU")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(current_dir, f'runs_semantic/run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    num_epochs = 100
    batch_size = 18
    accumulation_steps = 2
    crop_size = 512

    model = models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1').to(device)
    model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda')

    train_dataset = CityscapesDataset(train_img_dir, train_label_dir, crop_size=crop_size, batch_size=batch_size)
    val_dataset = CityscapesDataset(val_img_dir, val_label_dir, crop_size=crop_size, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    best_miou = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"第 [{epoch + 1}/{num_epochs}] 轮 训练中")

        for i, (imgs, labels) in enumerate(train_progress):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(imgs)['out']
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * accumulation_steps
            avg_loss = running_loss / (i + 1)
            train_progress.set_postfix({'训练损失': f'{avg_loss:.4f}'})

        train_loss = running_loss / len(train_loader)
        print(f"第 [{epoch + 1}/{num_epochs}] 轮 - 训练损失: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        val_progress = tqdm(val_loader, desc="验证中")

        with torch.no_grad():
            for imgs, labels in val_progress:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)['out']
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                miou = calculate_miou(outputs, labels, num_classes=19)
                val_miou += miou

        val_loss /= len(val_loader)
        val_miou /= len(val_loader)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('mIoU/Val', val_miou, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), os.path.join(current_dir, "models/best_semantic_model.pth"))
            print(f"第 {epoch + 1} 轮保存了新的最佳模型，验证损失: {val_loss:.4f}, mIoU: {val_miou:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(current_dir, f"models/semantic_model_epoch_{epoch + 1}.pth"))

        print(f"第 [{epoch + 1}/{num_epochs}] 轮 - 验证损失: {val_loss:.4f}, mIoU: {val_miou:.4f}, "
              f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

    writer.close()
    torch.save(model.state_dict(), os.path.join(current_dir, "models/final_semantic_model.pth"))
    print("语义分割训练完成！最终模型保存为 models/final_semantic_model.pth")


if __name__ == "__main__":
    os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)
    train_semantic_segmentation()