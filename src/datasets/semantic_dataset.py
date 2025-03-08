import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import torch
import random


class CityscapesDataset(Dataset):
    def __init__(self, img_dir, label_dir=None, crop_size=512, transform=None, test_mode=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.crop_size = crop_size
        self.test_mode = test_mode

        # 增强数据预处理
        if transform:
            self.transform = transform
        else:
            if not test_mode:
                self.transform = transforms.Compose([
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        self.images = []
        self.labels = []

        for city in tqdm(os.listdir(img_dir), desc="扫描城市"):
            city_img_dir = os.path.join(img_dir, city)
            if os.path.isdir(city_img_dir):
                for img_file in os.listdir(city_img_dir):
                    if img_file.endswith('leftImg8bit.png'):
                        self.images.append(os.path.join(city_img_dir, img_file))
                        if label_dir:
                            label_file = img_file.replace('leftImg8bit.png', 'gtFine_labelIds.png')
                            self.labels.append(os.path.join(label_dir, city, label_file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')

        if self.label_dir:
            label = Image.open(self.labels[idx])

            # 应用相同的随机变换
            if not self.test_mode and random.random() > 0.5:
                # 随机缩放 (0.75-1.25)
                scale = random.uniform(0.75, 1.25)
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.BILINEAR)
                label = label.resize((new_width, new_height), Image.NEAREST)

            # 随机裁剪
            if not self.test_mode and self.crop_size and img.width > self.crop_size and img.height > self.crop_size:
                top = random.randint(0, img.height - self.crop_size)
                left = random.randint(0, img.width - self.crop_size)
                img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
                label = label.crop((left, top, left + self.crop_size, top + self.crop_size))

            # 应用颜色增强和归一化
            img_tensor = self.transform(img)

            # 处理标签
            label_np = np.array(label, dtype=np.uint8)
            id2trainId = np.full((256,), 255, dtype=np.uint8)
            id2trainId[0] = 0  # unlabeled
            id2trainId[1] = 0  # ego vehicle
            id2trainId[2] = 0  # rectification border
            id2trainId[3] = 0  # out of roi
            id2trainId[4] = 0  # static
            id2trainId[5] = 0  # dynamic
            id2trainId[6] = 0  # ground
            id2trainId[7] = 0  # road
            id2trainId[8] = 1  # sidewalk
            id2trainId[9] = 2  # building
            id2trainId[10] = 3  # wall
            id2trainId[11] = 4  # fence
            id2trainId[12] = 5  # pole
            id2trainId[13] = 6  # traffic light
            id2trainId[14] = 7  # traffic sign
            id2trainId[15] = 8  # vegetation
            id2trainId[16] = 9  # terrain
            id2trainId[17] = 10  # sky
            id2trainId[18] = 11  # person
            id2trainId[19] = 12  # rider
            id2trainId[20] = 13  # car
            id2trainId[21] = 14  # truck
            id2trainId[22] = 15  # bus
            id2trainId[23] = 16  # train
            id2trainId[24] = 17  # motorcycle
            id2trainId[25] = 18  # bicycle
            label_np = id2trainId[label_np]

            label_tensor = torch.from_numpy(label_np).long()
            return img_tensor, label_tensor, os.path.basename(self.images[idx])

        # 测试模式或无标签情况
        img_tensor = self.transform(img)
        return img_tensor, os.path.basename(self.images[idx])