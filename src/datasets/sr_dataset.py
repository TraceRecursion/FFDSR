import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, crop_size=None, preload=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.crop_size = crop_size
        self.preload = preload
        self.to_tensor = transforms.ToTensor()
        self.hr_files = []
        self.lr_files = []

        hr_dict = {f.replace('leftImg8bit.png', ''): os.path.join(self.hr_dir, city, f)
                   for city in os.listdir(hr_dir) if os.path.isdir(os.path.join(hr_dir, city))
                   for f in os.listdir(os.path.join(hr_dir, city)) if f.endswith('leftImg8bit.png')}
        lr_dict = {f.replace('leftImg8bit_lr_x4.png', ''): os.path.join(self.lr_dir, city, f)
                   for city in os.listdir(lr_dir) if os.path.isdir(os.path.join(lr_dir, city))
                   for f in os.listdir(os.path.join(lr_dir, city)) if f.endswith('leftImg8bit_lr_x4.png')}

        for key in hr_dict:
            if key in lr_dict:
                self.hr_files.append(hr_dict[key])
                self.lr_files.append(lr_dict[key])

        if self.preload:
            self.hr_data = [self.to_tensor(Image.open(f).convert('RGB')) for f in
                            tqdm(self.hr_files, desc="加载HR数据")]
            self.lr_data = [self.to_tensor(Image.open(f).convert('RGB')) for f in
                            tqdm(self.lr_files, desc="加载LR数据")]

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        if self.preload:
            hr_img, lr_img = self.hr_data[idx], self.lr_data[idx]
        else:
            hr_img = self.to_tensor(Image.open(self.hr_files[idx]).convert('RGB'))
            lr_img = self.to_tensor(Image.open(self.lr_files[idx]).convert('RGB'))

        # 确保HR和LR图像尺寸比例为4:1
        if self.crop_size and hr_img.shape[1] > self.crop_size and hr_img.shape[2] > self.crop_size:
            # 随机选择一个HR裁剪区域
            y = np.random.randint(0, hr_img.shape[1] - self.crop_size)
            x = np.random.randint(0, hr_img.shape[2] - self.crop_size)
            # 对HR进行裁剪
            hr_img = hr_img[:, y:y + self.crop_size, x:x + self.crop_size]
            # 确保LR裁剪区域与HR对应
            lr_crop_size = self.crop_size // 4
            lr_y, lr_x = y // 4, x // 4
            lr_img = lr_img[:, lr_y:lr_y + lr_crop_size, lr_x:lr_x + lr_crop_size]

        return lr_img, hr_img