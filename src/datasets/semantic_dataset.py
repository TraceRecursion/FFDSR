import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import torch


class CityscapesDataset(Dataset):
    def __init__(self, img_dir, label_dir=None, crop_size=512, transform=None, test_mode=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.crop_size = crop_size
        self.test_mode = test_mode
        self.transform = transform or transforms.Compose([
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
        img_tensor = self.transform(img)

        if self.label_dir:
            label = Image.open(self.labels[idx])
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

            if not self.test_mode and self.crop_size and img_tensor.shape[1] > self.crop_size:
                top = np.random.randint(0, img_tensor.shape[1] - self.crop_size)
                left = np.random.randint(0, img_tensor.shape[2] - self.crop_size)
                img_tensor = img_tensor[:, top:top + self.crop_size, left:left + self.crop_size]
                label_np = label_np[top:top + self.crop_size, left:left + self.crop_size]
            label_tensor = torch.from_numpy(label_np).long()
            return img_tensor, label_tensor, os.path.basename(self.images[idx])
        return img_tensor, os.path.basename(self.images[idx])