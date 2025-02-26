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

# 数据缓存目录
cache_dir = os.path.join(current_dir, 'data_cache')
os.makedirs(cache_dir, exist_ok=True)


# 1. 数据准备
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, crop_size=None, use_cache=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.crop_size = crop_size
        self.use_cache = use_cache
        self.hr_files = sorted(os.listdir(self.hr_dir))
        self.lr_files = sorted(os.listdir(self.lr_dir))

        if len(self.hr_files) != len(self.lr_files):
            raise ValueError(f"HR和LR文件数量不匹配: {len(self.hr_files)} vs {len(self.lr_files)}")

        hr_cache_name = f'{os.path.basename(os.path.dirname(hr_dir))}_{os.path.basename(hr_dir)}_cache.pt'
        lr_cache_name = f'{os.path.basename(os.path.dirname(lr_dir))}_{os.path.basename(lr_dir)}_cache.pt'
        self.cache_hr = os.path.join(cache_dir, hr_cache_name)
        self.cache_lr = os.path.join(cache_dir, lr_cache_name)

        if self.use_cache and os.path.exists(self.cache_hr) and os.path.exists(self.cache_lr):
            print(f"加载缓存数据: {self.cache_hr}, {self.cache_lr}")
            self.hr_data = torch.load(self.cache_hr)
            self.lr_data = torch.load(self.cache_lr)
            if len(self.hr_data) != len(self.lr_data):
                raise ValueError(f"缓存中HR和LR数据长度不匹配: {len(self.hr_data)} vs {len(self.lr_data)}")
        else:
            print(f"缓存不存在，正在生成缓存: {self.cache_hr}, {self.cache_lr}")
            self.hr_data = []
            self.lr_data = []
            to_tensor = transforms.ToTensor()
            for hr_file, lr_file in tqdm(zip(self.hr_files, self.lr_files), total=len(self.hr_files),
                                         desc="缓存数据预处理"):
                hr_path = os.path.join(self.hr_dir, hr_file)
                lr_path = os.path.join(self.lr_dir, lr_file)
                try:
                    hr_img = Image.open(hr_path).convert('RGB')
                    lr_img = Image.open(lr_path).convert('RGB')
                    self.hr_data.append(to_tensor(hr_img))
                    self.lr_data.append(to_tensor(lr_img))
                except Exception as e:
                    print(f"跳过文件 {hr_file}/{lr_file}，错误: {e}")
            torch.save(self.hr_data, self.cache_hr)
            torch.save(self.lr_data, self.cache_lr)
            print(f"缓存已保存至: {self.cache_hr}, {self.cache_lr}")

    def __len__(self):
        return min(len(self.hr_data), len(self.lr_data))

    def __getitem__(self, idx):
        if idx >= len(self.hr_data) or idx >= len(self.lr_data):
            raise IndexError(f"索引 {idx} 超出范围: HR {len(self.hr_data)}, LR {len(self.lr_data)}")
        hr_img = self.hr_data[idx]
        lr_img = self.lr_data[idx]

        if self.crop_size:
            hr_h, hr_w = hr_img.shape[1:]
            max_x = hr_w - self.crop_size
            max_y = hr_h - self.crop_size
            if max_x > 0 and max_y > 0:
                x = np.random.randint(0, max_x)
                y = np.random.randint(0, max_y)
                hr_img = hr_img[:, y:y + self.crop_size, x:x + self.crop_size]
                lr_img = lr_img[:, y // 4:(y + self.crop_size) // 4, x // 4:(x + self.crop_size) // 4]

        return lr_img, hr_img


# 2. 模型设计（保持不变）
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
        return residual + 0.1 * out


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
        self.att1 = CBAM(64)
        self.conv3 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.up2 = nn.PixelShuffle(2)
        self.conv_up2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.att2 = CBAM(64)
        self.fusion = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.body(x)
        x2 = self.conv2(x)
        x2 = self.up1(x2)
        x2 = self.conv_up1(x2)
        x2 = self.att1(x2)
        x3 = self.conv3(x2)
        x3 = self.up2(x3)
        x3 = self.conv_up2(x3)
        x3 = self.att2(x3)
        x_fused = self.fusion(
            torch.cat([F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False), x3], dim=1))
        x = self.conv_out(x_fused)
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

        for param in self.semantic_model.backbone.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False

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
        semantic_feature = F.interpolate(semantic_feature, size=(target_h, target_w), mode='bilinear',
                                         align_corners=False)
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


# 3. 感知损失（修复尺寸不匹配问题）
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights='DEFAULT').features.eval().to(device)
        self.vgg = vgg
        self.layer = '3'
        self.semantic_model = deeplabv3_resnet101(weights='DEFAULT').eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        for param in self.semantic_model.parameters():
            param.requires_grad = False

    def forward(self, sr_img, hr_img, lr_img, compute_semantic=True):
        sr_feat = self.vgg(sr_img)
        hr_feat = self.vgg(hr_img)
        perc_loss = F.mse_loss(sr_feat, hr_feat)

        if compute_semantic:
            sr_img_norm = (sr_img - self.mean) / self.std
            lr_img_norm = (lr_img - self.mean) / self.std
            sr_semantic = self.semantic_model(sr_img_norm)['out']
            lr_semantic = self.semantic_model(lr_img_norm)['out']
            # 将 sr_semantic 下采样到 lr_semantic 的分辨率
            sr_semantic = F.interpolate(sr_semantic, size=lr_semantic.shape[2:], mode='bilinear', align_corners=False)
            if torch.any(torch.isnan(sr_semantic)) or torch.any(torch.isnan(lr_semantic)):
                print("语义输出中检测到 NaN！")
                print(f"SR Semantic: 最小值={sr_semantic.min().item():.4f}, 最大值={sr_semantic.max().item():.4f}")
                print(f"LR Semantic: 最小值={lr_semantic.min().item():.4f}, 最大值={lr_semantic.max().item():.4f}")
            sr_semantic = torch.clamp(sr_semantic, -10, 10)
            lr_semantic = torch.clamp(lr_semantic, -10, 10)
            semantic_loss = F.mse_loss(sr_semantic, lr_semantic)
        else:
            semantic_loss = torch.tensor(0.0, device=sr_img.device)

        return perc_loss, semantic_loss


# 4. 训练与评估
def train_and_validate(model, train_loader, val_loader, criterion_l1, criterion_perceptual, optimizer, scheduler,
                       device, num_epochs=100, accumulation_steps=2):
    model.to(device)
    os.makedirs("models", exist_ok=True)
    scaler = GradScaler('cuda')
    best_val_loss = float('inf')
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"第 [{epoch + 1}/{num_epochs}] 轮 训练中")

        for i, (lr_img, hr_img) in enumerate(progress_bar):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            with autocast('cuda'):
                sr_img = model(lr_img)
                l1_loss = criterion_l1(sr_img, hr_img)
                compute_semantic = (i % 5 == 0)
                perc_loss, semantic_loss = criterion_perceptual(sr_img, hr_img, lr_img, compute_semantic)
                loss = (2.0 * l1_loss + 0.5 * perc_loss + 0.00005 * semantic_loss) / accumulation_steps

                if i == 0:
                    print(f"第 {epoch + 1} 轮, 第 {i + 1} 批次:")
                    print(f"低分辨率图像: 最小值={lr_img.min().item():.4f}, 最大值={lr_img.max().item():.4f}")
                    print(f"高分辨率图像: 最小值={hr_img.min().item():.4f}, 最大值={hr_img.max().item():.4f}")
                    print(
                        f"超分辨率图像: 最小值={sr_img.min().item():.4f}, 最大值={sr_img.max().item():.4f}, "
                        f"平均值={sr_img.mean().item():.4f}, 标准差={sr_img.std().item():.4f}"
                    )
                    print(
                        f"L1 Loss: {l1_loss.item():.4f}, Perc Loss: {perc_loss.item():.4f}, "
                        f"Semantic Loss: {semantic_loss.item():.4f}, 总损失: {loss.item() * accumulation_steps:.4f}"
                    )

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            avg_loss = running_loss / (i + 1)
            progress_bar.set_postfix({'训练损失': f'{avg_loss:.4f}'})

        train_loss = running_loss / len(train_loader)

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
                    perc_loss, semantic_loss = criterion_perceptual(sr_img, hr_img, lr_img, compute_semantic=True)
                    loss = 2.0 * l1_loss + 0.5 * perc_loss + 0.00005 * semantic_loss

                val_loss += loss.item()
                val_psnr += psnr(sr_img, hr_img).item()
                val_ssim += ssim(sr_img, hr_img).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print(
                f"第 {epoch + 1} 轮保存了新的最佳模型，验证损失: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}"
            )

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"models/model_epoch_{epoch + 1}.pth")

        print(
            f"第 [{epoch + 1}/{num_epochs}] 轮 - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
            f"PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}"
        )

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

    # 验证数据集完整性
    print(f"训练集 HR 文件数: {len(os.listdir(train_hr_dir))}")
    print(f"训练集 LR 文件数: {len(os.listdir(train_lr_dir))}")
    print(f"验证集 HR 文件数: {len(os.listdir(val_hr_dir))}")
    print(f"验证集 LR 文件数: {len(os.listdir(val_lr_dir))}")

    # 数据集加载到内存并使用缓存
    train_dataset = SRDataset(hr_dir=train_hr_dir, lr_dir=train_lr_dir, crop_size=512, use_cache=True)
    val_dataset = SRDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir, crop_size=512, use_cache=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)

    model = FeatureFusionSR().to(device)
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_and_validate(model, train_loader, val_loader, criterion_l1, criterion_perceptual, optimizer, scheduler,
                       device, num_epochs=100, accumulation_steps=2)

    torch.save(model.state_dict(), "models/final_model.pth")
    print("最终模型保存为 models/final_model.pth")