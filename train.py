import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import resnet18
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import time

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 数据集基础路径
base_data_dir = os.path.join(current_dir, '../../Documents/数据集')

train_hr_dir = os.path.join(base_data_dir, 'DIV2K/train/DIV2K_train_HR')
train_lr_dir = os.path.join(base_data_dir, 'DIV2K/train/DIV2K_train_LR_bicubic/X4')
val_hr_dir = os.path.join(base_data_dir, 'DIV2K/val/DIV2K_valid_HR')
val_lr_dir = os.path.join(base_data_dir, 'DIV2K/val/DIV2K_valid_LR_bicubic/X4')

# 修改导入方式
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# 1. 数据准备（不变）
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, crop_size=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(os.listdir(self.hr_dir))

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, sorted(os.listdir(self.hr_dir))[idx])
        lr_path = os.path.join(self.lr_dir, sorted(os.listdir(self.lr_dir))[idx])

        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')

        if self.crop_size:
            hr_w, hr_h = hr_img.size
            lr_w, lr_h = lr_img.size
            max_x = hr_w - self.crop_size
            max_y = hr_h - self.crop_size
            if max_x > 0 and max_y > 0:
                x = np.random.randint(0, max_x)
                y = np.random.randint(0, max_y)
                hr_img = hr_img.crop((x, y, x + self.crop_size, y + self.crop_size))
                lr_img = lr_img.crop((x // 4, y // 4, (x + self.crop_size) // 4, (y + self.crop_size) // 4))

        hr_img = self.to_tensor(hr_img)
        lr_img = self.to_tensor(lr_img)

        return lr_img, hr_img

# 2. 模型设计（不变）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EDSR(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, num_blocks=8, upscale_factor=4):
        super(EDSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_blocks)]
        )
        self.conv2 = nn.Conv2d(64, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.body(x)
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        return x

class FeatureFusionSR(nn.Module):
    def __init__(self):
        super(FeatureFusionSR, self).__init__()
        self.semantic_model = deeplabv3_resnet101(weights='DEFAULT').eval()
        self.embedding = nn.Embedding(19, 64)

        self.resnet = resnet18(weights='DEFAULT')
        self.resnet_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.resnet_bn1 = self.resnet.bn1
        self.resnet_relu = self.resnet.relu
        self.resnet_layer1 = self.resnet.layer1

        self.se_block = SEBlock(128)
        self.fusion_conv = nn.Conv2d(128, 256, kernel_size=1)

        self.sr_net = EDSR()

    def forward(self, lr_img):
        with torch.no_grad():
            semantic_output = self.semantic_model(lr_img)['out']
        semantic_labels = semantic_output.argmax(dim=1)
        semantic_feature = self.embedding(semantic_labels).permute(0, 3, 1, 2)

        x = self.resnet_conv1(lr_img)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        low_level_feature = self.resnet_layer1(x)

        target_h, target_w = low_level_feature.shape[2:]
        semantic_feature = F.interpolate(
            semantic_feature, size=(target_h, target_w), mode='bilinear', align_corners=False
        )

        fused_feature = torch.cat([semantic_feature, low_level_feature], dim=1)
        fused_feature = self.se_block(fused_feature)
        fused_feature = self.fusion_conv(fused_feature)

        sr_img = self.sr_net(fused_feature)
        return sr_img

# 3. 感知损失定义
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights='DEFAULT').features.eval()
        self.vgg = vgg.to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = {'3': 64, '8': 128, '17': 256}  # conv1_2, conv2_2, conv3_4

    def forward(self, sr_img, hr_img):
        loss = 0.0
        for name, module in self.vgg.named_children():
            sr_img = module(sr_img)
            hr_img = module(hr_img)
            if name in self.layers:
                loss += F.mse_loss(sr_img, hr_img)
            if name == '17':  # 到 conv3_4 停止
                break
        return loss

# 4. 训练与评估
def train_and_validate(model, train_loader, val_loader, criterion_l1, criterion_perceptual, optimizer, scheduler, device, num_epochs=100):
    model.to(device)
    os.makedirs("models", exist_ok=True)

    best_val_loss = float('inf')
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training")

        for lr_img, hr_img in progress_bar:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            optimizer.zero_grad()
            sr_img = model(lr_img)
            l1_loss = criterion_l1(sr_img, hr_img)
            perc_loss = criterion_perceptual(sr_img, hr_img)
            loss = l1_loss + 0.1 * perc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (len(progress_bar) + 1)
            progress_bar.set_postfix({'Train Loss': f'{avg_loss:.4f}'})

        train_loss = running_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                sr_img = model(lr_img)
                l1_loss = criterion_l1(sr_img, hr_img)
                perc_loss = criterion_perceptual(sr_img, hr_img)
                loss = l1_loss + 0.1 * perc_loss
                val_loss += loss.item()
                val_psnr += psnr(sr_img, hr_img).item()
                val_ssim += ssim(sr_img, hr_img).item()

        val_loss = val_loss / len(val_loader)
        val_psnr = val_psnr / len(val_loader)
        val_ssim = val_ssim / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"新最佳模型在 epoch {epoch + 1} 处保存, Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"models/model_epoch_{epoch + 1}.pth")

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

    print("训练完成！")

# 主函数
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用设备: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用设备: MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("没有可用的GPU，使用设备: CPU")

    # train_hr_dir = "/Users/sydg/Documents/数据集/DIV2K/train/DIV2K_train_HR"
    # train_lr_dir = "/Users/sydg/Documents/数据集/DIV2K/train/DIV2K_train_LR_bicubic/X4"
    # val_hr_dir = "/Users/sydg/Documents/数据集/DIV2K/val/DIV2K_valid_HR"
    # val_lr_dir = "/Users/sydg/Documents/数据集/DIV2K/val/DIV2K_valid_LR_bicubic/X4"

    train_dataset = SRDataset(hr_dir=train_hr_dir, lr_dir=train_lr_dir, crop_size=512)
    val_dataset = SRDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir, crop_size=512)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = FeatureFusionSR().to(device)
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_and_validate(model, train_loader, val_loader, criterion_l1, criterion_perceptual, optimizer, scheduler, device, num_epochs=100)

    torch.save(model.state_dict(), "models/final_model.pth")
    print("最终模型保存为 models/final_model.pth")