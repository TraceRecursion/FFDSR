import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(current_dir, '../../Documents/数据集')
train_hr_dir = os.path.join(base_data_dir, 'DIV2K/train/DIV2K_train_HR')
train_lr_dir = os.path.join(base_data_dir, 'DIV2K/train/DIV2K_train_LR_bicubic/X4')
val_hr_dir = os.path.join(base_data_dir, 'DIV2K/val/DIV2K_valid_HR')
val_lr_dir = os.path.join(base_data_dir, 'DIV2K/val/DIV2K_valid_LR_bicubic/X4')

# 1. 数据准备
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

# 2. 模型设计
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return self.sigmoid(avg_out + max_out).view(b, c, 1, 1) * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return residual + 0.1 * out  # 添加残差缩放因子

class EnhancedEDSR(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, num_blocks=64):
        super(EnhancedEDSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(64) for _ in range(num_blocks)]
        )
        self.conv2 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.up1 = nn.PixelShuffle(2)
        self.conv_up1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.up2 = nn.PixelShuffle(2)
        self.conv_up2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.body(x)
        x = self.conv2(x)
        x = self.up1(x)
        x = self.conv_up1(x)
        x = self.conv3(x)
        x = self.up2(x)
        x = self.conv_up2(x)
        x = self.conv_out(x)
        return torch.sigmoid(x)

class FeatureFusionSR(nn.Module):
    def __init__(self):
        super(FeatureFusionSR, self).__init__()
        self.semantic_model = deeplabv3_resnet101(weights='DEFAULT')
        self.embedding = nn.Embedding(21, 64)
        self.resnet = models.resnet50(weights='DEFAULT')
        self.resnet_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.resnet_bn1 = self.resnet.bn1
        self.resnet_relu = self.resnet.relu
        self.resnet_maxpool = nn.Identity()
        self.resnet_layer1 = self.resnet.layer1
        self.resnet_layer2 = self.resnet.layer2
        self.resnet_layer3 = self.resnet.layer3
        self.fusion_conv = nn.Conv2d(1856, 256, kernel_size=1)
        self.cbam = CBAM(256)
        self.sr_net = EnhancedEDSR()

    def forward_semantic(self, lr_img):
        semantic_output = self.semantic_model(lr_img)['out']
        semantic_labels = semantic_output.argmax(dim=1)
        semantic_feature = self.embedding(semantic_labels).permute(0, 3, 1, 2)
        return semantic_feature

    def forward_low_level(self, lr_img):
        x = self.resnet_conv1(lr_img)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)
        layer1_out = self.resnet_layer1(x)
        layer2_out = self.resnet_layer2(layer1_out)
        layer3_out = self.resnet_layer3(layer2_out)
        layer2_out = F.interpolate(layer2_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False)
        layer3_out = F.interpolate(layer3_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False)
        low_level_feature = torch.cat([layer1_out, layer2_out, layer3_out], dim=1)
        return low_level_feature

    def forward_fusion_and_sr(self, semantic_feature, low_level_feature):
        target_h, target_w = low_level_feature.shape[2:]
        semantic_feature = F.interpolate(semantic_feature, size=(target_h, target_w), mode='bilinear', align_corners=False)
        fused_feature = torch.cat([semantic_feature, low_level_feature], dim=1)
        fused_feature = self.fusion_conv(fused_feature)
        fused_feature = self.cbam(fused_feature)
        sr_img = self.sr_net(fused_feature)
        return sr_img

    def forward(self, lr_img):
        semantic_feature = self.forward_semantic(lr_img)
        low_level_feature = self.forward_low_level(lr_img)
        sr_img = self.forward_fusion_and_sr(semantic_feature, low_level_feature)
        return sr_img

# 3. 感知损失
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights='DEFAULT').features.eval().to(device)
        self.vgg = vgg
        self.layer = '3'
        self.semantic_model = deeplabv3_resnet101(weights='DEFAULT').eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        for param in self.semantic_model.parameters():
            param.requires_grad = False

    def forward(self, sr_img, hr_img, lr_img):
        sr_feat = self.vgg(sr_img)
        hr_feat = self.vgg(hr_img)
        perc_loss = F.mse_loss(sr_feat, hr_feat)

        sr_semantic = self.semantic_model(sr_img)['out']
        lr_semantic = self.semantic_model(lr_img)['out']
        sr_semantic = F.interpolate(sr_semantic, size=lr_semantic.shape[2:], mode='bilinear', align_corners=False)

        # 调试输出
        if torch.any(torch.isnan(sr_semantic)) or torch.any(torch.isnan(lr_semantic)):
            print("语义输出中检测到 NaN！")
            print(f"SR Semantic before norm: min={sr_semantic.min().item():.4f}, max={sr_semantic.max().item():.4f}")
            print(f"LR Semantic before norm: min={lr_semantic.min().item():.4f}, max={lr_semantic.max().item():.4f}")

        # 改进标准化：限制logits范围
        sr_semantic = torch.tanh(sr_semantic)  # 限制到[-1, 1]
        lr_semantic = torch.tanh(lr_semantic)
        sr_semantic = (sr_semantic - sr_semantic.mean()) / (sr_semantic.std() + 1e-8)
        lr_semantic = (lr_semantic - lr_semantic.mean()) / (lr_semantic.std() + 1e-8)

        semantic_loss = F.kl_div(F.log_softmax(sr_semantic, dim=1), F.softmax(lr_semantic, dim=1), reduction='batchmean')
        return perc_loss, semantic_loss

# 4. 训练与评估
def train_and_validate(model, train_loader, val_loader, criterion_l1, criterion_perceptual, optimizer, scheduler, device, num_epochs=100):
    model.to(device)
    os.makedirs("models", exist_ok=True)
    scaler = GradScaler('cuda')
    best_val_loss = float('inf')
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training")

        for i, (lr_img, hr_img) in enumerate(progress_bar):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                sr_img = model(lr_img)
                if i == 0:
                    print(f"Epoch {epoch + 1}, Batch {i + 1}:")
                    print(f"LR Img: min={lr_img.min().item():.4f}, max={lr_img.max().item():.4f}")
                    print(f"HR Img: min={hr_img.min().item():.4f}, max={hr_img.max().item():.4f}")
                    print(f"SR Img: min={sr_img.min().item():.4f}, max={sr_img.max().item():.4f}, mean={sr_img.mean().item():.4f}, std={sr_img.std().item():.4f}")

                l1_loss = criterion_l1(sr_img, hr_img)
                perc_loss, semantic_loss = criterion_perceptual(sr_img, hr_img, lr_img)
                loss = l1_loss + 0.1 * perc_loss + 0.0001 * semantic_loss

                if i == 0:
                    print(f"L1 Loss: {l1_loss.item():.4f}, Perc Loss: {perc_loss.item():.4f}, Semantic Loss: {semantic_loss.item():.4f}, Total Loss: {loss.item():.4f}")

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 增强梯度裁剪
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
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
                with autocast('cuda'):
                    sr_img = model(lr_img)
                    l1_loss = criterion_l1(sr_img, hr_img)
                    perc_loss, semantic_loss = criterion_perceptual(sr_img, hr_img, lr_img)
                    loss = l1_loss + 0.1 * perc_loss + 0.0001 * semantic_loss

                val_loss += loss.item()
                val_psnr += psnr(sr_img, hr_img).item()
                val_ssim += ssim(sr_img, hr_img).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"New best model saved at epoch {epoch + 1}, Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"models/model_epoch_{epoch + 1}.pth")

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")

    print("Training completed!")

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

    train_dataset = SRDataset(hr_dir=train_hr_dir, lr_dir=train_lr_dir, crop_size=512)
    val_dataset = SRDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir, crop_size=512)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model = FeatureFusionSR().to(device)
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # 调整学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_and_validate(model, train_loader, val_loader, criterion_l1, criterion_perceptual, optimizer, scheduler, device, num_epochs=100)

    torch.save(model.state_dict(), "models/final_model.pth")
    print("最终模型保存为 models/final_model.pth")