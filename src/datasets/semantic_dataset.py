import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class CityscapesDataset(Dataset):
    def __init__(self, img_dir, label_dir=None, crop_size=512, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.crop_size = crop_size
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
            if self.crop_size and img_tensor.shape[1] > self.crop_size:
                top = np.random.randint(0, img_tensor.shape[1] - self.crop_size)
                left = np.random.randint(0, img_tensor.shape[2] - self.crop_size)
                img_tensor = img_tensor[:, top:top + self.crop_size, left:left + self.crop_size]
                label_np = label_np[top:top + self.crop_size, left:left + self.crop_size]
            return img_tensor, label_np, os.path.basename(self.images[idx])
        return img_tensor, os.path.basename(self.images[idx])