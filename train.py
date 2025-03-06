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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision.utils import save_image  # 新增：用于保存图像调试

# 获取当前文件所在目录并设置相对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(current_dir, '../../Documents/数据集/CitySpaces/leftImg8bit_trainvaltest/leftImg8bit')
train_hr_dir = os.path.join(base_data_dir, 'train')
train_lr_dir = os.path.join(base_data_dir, 'train_lr_x4')
val_hr_dir = os.path.join(base_data_dir, 'val')
val_lr_dir = os.path.join(base_data_dir, 'val_lr_x4')
test_hr_dir = os.path.join(base_data_dir, 'test')
test_lr_dir = os.path.join(base_data_dir, 'test_lr_x4')


# 1. 数据准备
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, crop_size=None, preload=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.crop_size = crop_size
        self.preload = preload

        if not os.path.exists(self.hr_dir):
            raise FileNotFoundError(f"HR 目录不存在: {self.hr_dir}")
        if not os.path.exists(self.lr_dir):
            raise FileNotFoundError(f"LR 目录不存在: {self.lr_dir}")

        hr_dict = {}
        lr_dict = {}
        for city in os.listdir(self.hr_dir):
            city_hr_path = os.path.join(self.hr_dir, city)
            if os.path.isdir(city_hr_path):
                for f in os.listdir(city_hr_path):
                    if f.endswith('leftImg8bit.png'):
                        key = f.replace('leftImg8bit.png', '')
                        hr_dict[key] = os.path.join(city_hr_path, f)
        for city in os.listdir(self.lr_dir):
            city_lr_path = os.path.join(self.lr_dir, city)
            if os.path.isdir(city_lr_path):
                for f in os.listdir(city_lr_path):
                    if f.endswith('leftImg8bit_lr_x4.png'):
                        key = f.replace('leftImg8bit_lr_x4.png', '')
                        lr_dict[key] = os.path.join(city_lr_path, f)

        self.hr_files = []
        self.lr_files = []
        for key in hr_dict:
            if key in lr_dict:
                self.hr_files.append(hr_dict[key])
                self.lr_files.append(lr_dict[key])
            else:
                print(f"警告: HR 文件 {hr_dict[key]} 未找到对应的 LR 文件")

        print(f"HR 文件总数: {len(self.hr_files)}")
        print(f"LR 文件总数: {len(self.lr_files)}")
        if len(self.hr_files) != len(self.lr_files):
            raise ValueError(f"HR和LR文件数量不匹配: {len(self.hr_files)} vs {len(self.lr_files)}")
        if len(self.hr_files) == 0:
            raise ValueError("未找到任何匹配的 HR 和 LR 文件对")

        self.to_tensor = transforms.ToTensor()

        if self.preload:
            print(f"预加载数据集到内存: {self.hr_dir}")
            self.hr_data = []
            self.lr_data = []
            total_size_mb = 0
            for hr_path, lr_path in tqdm(zip(self.hr_files, self.lr_files), total=len(self.hr_files), desc="加载数据"):
                hr_img = Image.open(hr_path).convert('RGB')
                lr_img = Image.open(lr_path).convert('RGB')
                hr_tensor = self.to_tensor(hr_img)
                lr_tensor = self.to_tensor(lr_img)
                self.hr_data.append(hr_tensor)
                self.lr_data.append(lr_tensor)
                total_size_mb += (hr_tensor.element_size() * hr_tensor.nelement() +
                                  lr_tensor.element_size() * lr_tensor.nelement()) / 1024 ** 2
            print(f"数据预加载完成，总内存占用约 {total_size_mb:.2f} MB")
        else:
            self.hr_data = None
            self.lr_data = None

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        if self.preload:
            hr_img = self.hr_data[idx]
            lr_img = self.lr_data[idx]
        else:
            hr_path = self.hr_files[idx]
            lr_path = self.lr_files[idx]
            hr_img = Image.open(hr_path).convert('RGB')
            lr_img = Image.open(lr_path).convert('RGB')
            hr_img = self.to_tensor(hr_img)
            lr_img = self.to_tensor(lr_img)

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
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return residual + self.scale * out


class EnhancedEDSR(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, num_blocks=32, use_sigmoid=True):  # 新增：use_sigmoid 参数
        super(EnhancedEDSR, self).__init__()
        self.use_sigmoid = use_sigmoid  # 新增：控制是否使用 sigmoid
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(64) for _ in range(num_blocks)]
        )
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.att1 = CBAM(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
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
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = self.conv_up1(x2)
        x2 = self.att1(x2)
        x3 = self.conv3(x2)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = self.conv_up2(x3)
        x3 = self.att2(x3)
        x_fused = self.fusion(
            torch.cat([F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False), x3], dim=1))
        x = self.conv_out(x_fused)
        if self.use_sigmoid:  # 新增：根据参数决定是否应用 sigmoid
            x = torch.sigmoid(x)
        return x


class FeatureFusionSR(nn.Module):
    def __init__(self, semantic_model_path=None):
        super(FeatureFusionSR, self).__init__()
        self.semantic_model = deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
        self.semantic_model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
        self.semantic_model.aux_classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
        if semantic_model_path:
            semantic_state_dict = torch.load(semantic_model_path, map_location='cpu')
            self.semantic_model.load_state_dict(semantic_state_dict)
            print(f"加载预训练语义模型: {semantic_model_path}")
        self.semantic_model.eval()
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.embedding = nn.Embedding(19, 64)
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

        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet_layer1.parameters():
            param.requires_grad = False
        for param in self.resnet_layer2.parameters():
            param.requires_grad = False

    def forward_semantic(self, lr_img):
        with torch.no_grad():
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


# 3. 感知损失
class PerceptualLoss(nn.Module):
    def __init__(self, device, semantic_model_path=None):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights='DEFAULT').features.eval().to(device)
        self.vgg = vgg
        self.layer = '3'
        self.semantic_model = deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
        self.semantic_model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
        self.semantic_model.aux_classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
        if semantic_model_path:
            semantic_state_dict = torch.load(semantic_model_path, map_location=device)
            self.semantic_model.load_state_dict(semantic_state_dict)
            print(f"感知损失加载预训练语义模型: {semantic_model_path}")
        self.semantic_model.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        for param in self.semantic_model.parameters():
            param.requires_grad = False

    def forward(self, sr_img, hr_img, lr_img, compute_semantic=True, writer=None, global_step=None):
        sr_feat = self.vgg(sr_img)
        hr_feat = self.vgg(hr_img)
        perc_loss = F.mse_loss(sr_feat, hr_feat)

        if compute_semantic:
            with torch.no_grad():
                sr_img_norm = (sr_img - self.mean) / self.std
                lr_img_norm = (lr_img - self.mean) / self.std
                sr_semantic = self.semantic_model(sr_img_norm)['out']
                lr_semantic = self.semantic_model(lr_img_norm)['out']
            sr_semantic = F.interpolate(sr_semantic, size=lr_semantic.shape[2:], mode='bilinear', align_corners=False)
            sr_semantic = F.softmax(sr_semantic, dim=1)
            lr_semantic = F.softmax(lr_semantic, dim=1)
            semantic_loss = F.mse_loss(sr_semantic, lr_semantic)
            if torch.any(torch.isnan(semantic_loss)):
                print("语义损失中检测到 NaN！")
                print(f"SR Semantic: 最小值={sr_semantic.min().item():.4f}, 最大值={sr_semantic.max().item():.4f}")
                print(f"LR Semantic: 最小值={lr_semantic.min().item():.4f}, 最大值={lr_semantic.max().item():.4f}")
            if writer is not None and global_step is not None:
                writer.add_scalar('Loss/Semantic', semantic_loss.item(), global_step)
        else:
            semantic_loss = torch.tensor(0.0, device=sr_img.device)

        return perc_loss, semantic_loss


def color_consistency_loss(sr_img, hr_img):
    sr_mean = sr_img.mean(dim=[2, 3])
    hr_mean = hr_img.mean(dim=[2, 3])
    sr_var = sr_img.var(dim=[2, 3])
    hr_var = hr_img.var(dim=[2, 3])
    mean_loss = F.mse_loss(sr_mean, hr_mean)
    var_loss = F.mse_loss(sr_var, hr_var)
    return mean_loss + var_loss


# 4. 训练与评估
def train_and_validate(model, train_loader, val_loader, criterion_l1, criterion_perceptual, optimizer, scheduler,
                       device, log_dir, num_epochs=50, accumulation_steps=2):
    model.to(device)
    os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "debug_images"), exist_ok=True)  # 新增：保存调试图像的目录
    scaler = GradScaler('cuda')
    best_val_loss = float('inf')
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_semantic_loss = 0.0
        semantic_count = 0
        progress_bar = tqdm(train_loader, desc=f"第 [{epoch + 1}/{num_epochs}] 轮 训练中")

        for i, (lr_img, hr_img) in enumerate(progress_bar):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            global_step = epoch * len(train_loader) + i

            with autocast('cuda'):
                sr_img = model(lr_img)
                # 新增：检查 SR 和 HR 图像的值
                if i == 0:  # 只在每个 epoch 的第一个 batch 打印，避免过多输出
                    print(f"Epoch {epoch + 1}, Batch {i + 1}:")
                    print(
                        f"SR 图像 - Min: {sr_img.min().item():.4f}, Max: {sr_img.max().item():.4f}, Mean: {sr_img.mean().item():.4f}")
                    print(
                        f"HR 图像 - Min: {hr_img.min().item():.4f}, Max: {hr_img.max().item():.4f}, Mean: {hr_img.mean().item():.4f}")

                l1_loss = criterion_l1(sr_img, hr_img)
                compute_semantic = (i % 5 == 0)
                perc_loss, semantic_loss = criterion_perceptual(sr_img, hr_img, lr_img, compute_semantic, writer,
                                                                global_step)
                color_loss = color_consistency_loss(sr_img, hr_img)
                loss = (2.0 * l1_loss + 1.0 * perc_loss + 0.5 * semantic_loss + 0.1 * color_loss) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            avg_loss = running_loss / (i + 1)

            if compute_semantic:
                running_semantic_loss += semantic_loss.item()
                semantic_count += 1
                avg_semantic_loss = running_semantic_loss / semantic_count if semantic_count > 0 else 0.0
            else:
                avg_semantic_loss = running_semantic_loss / semantic_count if semantic_count > 0 else 0.0

            progress_bar.set_postfix({
                '训练损失': f'{avg_loss:.4f}',
                '语义损失': f'{avg_semantic_loss:.4f}'
            })

        train_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        first_batch = True
        with torch.no_grad():
            for lr_img, hr_img in tqdm(val_loader, desc="验证中"):
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                with autocast('cuda'):
                    sr_img = model(lr_img)
                    l1_loss = criterion_l1(sr_img, hr_img)
                    perc_loss, semantic_loss = criterion_perceptual(sr_img, hr_img, lr_img, compute_semantic=True,
                                                                    writer=None, global_step=None)
                    color_loss = color_consistency_loss(sr_img, hr_img)
                    loss = 2.0 * l1_loss + 1.0 * perc_loss + 0.5 * semantic_loss + 0.1 * color_loss

                val_loss += loss.item()
                val_psnr += psnr(sr_img, hr_img).item()
                val_ssim += ssim(sr_img, hr_img).item()

                if first_batch:
                    # 新增：检查 TensorBoard 输入并保存调试图像
                    print(f"Epoch {epoch + 1} - TensorBoard 输入检查:")
                    print(
                        f"SR_Image - Min: {sr_img[0].min().item():.4f}, Max: {sr_img[0].max().item():.4f}, Mean: {sr_img[0].mean().item():.4f}")
                    print(
                        f"HR_Image - Min: {hr_img[0].min().item():.4f}, Max: {hr_img[0].max().item():.4f}, Mean: {hr_img[0].mean().item():.4f}")
                    writer.add_image('SR_Image', sr_img[0], epoch)
                    writer.add_image('HR_Image', hr_img[0], epoch)
                    # 保存图像到本地以供手动检查
                    save_image(sr_img[0], os.path.join(log_dir, f"debug_images/sr_epoch_{epoch + 1}.png"))
                    save_image(hr_img[0], os.path.join(log_dir, f"debug_images/hr_epoch_{epoch + 1}.png"))
                    first_batch = False

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('PSNR/Val', val_psnr, epoch)
        writer.add_scalar('SSIM/Val', val_ssim, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "models/best_model.pth"))
            print(
                f"第 {epoch + 1} 轮保存了新的最佳模型，验证损失: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}"
            )

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f"models/model_epoch_{epoch + 1}.pth"))

        print(
            f"第 [{epoch + 1}/{num_epochs}] 轮 - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
            f"PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}"
        )

    writer.close()
    print("训练完成！")
    torch.save(model.state_dict(), os.path.join(log_dir, "models/final_model.pth"))
    print(f"最终模型保存为 {log_dir}/models/final_model.pth")


# 主函数
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(current_dir, f'runs/run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用设备: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用设备: MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("没有可用的GPU，使用设备: CPU")


    def count_files(directory, suffix):
        total = 0
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return 0
        for city in os.listdir(directory):
            city_path = os.path.join(directory, city)
            if os.path.isdir(city_path):
                total += len([f for f in os.listdir(city_path) if f.endswith(suffix)])
        return total


    print(f"训练集 HR 文件数: {count_files(train_hr_dir, 'leftImg8bit.png')}")
    print(f"训练集 LR 文件数: {count_files(train_lr_dir, 'leftImg8bit_lr_x4.png')}")
    print(f"验证集 HR 文件数: {count_files(val_hr_dir, 'leftImg8bit.png')}")
    print(f"验证集 LR 文件数: {count_files(val_lr_dir, 'leftImg8bit_lr_x4.png')}")

    train_dataset = SRDataset(hr_dir=train_hr_dir, lr_dir=train_lr_dir, crop_size=512, preload=True)
    val_dataset = SRDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir, crop_size=512, preload=True)
    train_loader = DataLoader(train_dataset, batch_size=13, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=13, shuffle=False, num_workers=8, pin_memory=True)

    semantic_model_path = os.path.join(current_dir, "models/best_semantic_model.pth")
    if not os.path.exists(semantic_model_path):
        raise FileNotFoundError(f"语义模型文件未找到: {semantic_model_path}")

    model = FeatureFusionSR(semantic_model_path=semantic_model_path).to(device)
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss(device, semantic_model_path=semantic_model_path)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_and_validate(model, train_loader, val_loader, criterion_l1, criterion_perceptual, optimizer, scheduler,
                       device, log_dir, num_epochs=50, accumulation_steps=2)