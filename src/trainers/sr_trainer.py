import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
from ..datasets.sr_dataset import SRDataset
from ..models.sr_model import (
    FeatureFusionSR, 
    FeatureFusionSR_NoSemantic,
    FeatureFusionSR_NoCBAM,
    FeatureFusionSR_SingleScale
)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from ..utils.common import get_device


class SRTrainer:
    def __init__(self, config):
        self.config = config
        self.device, device_name = get_device()
        print(f"使用设备: {device_name}")

        # 根据数据集样本确定模型的缩放比例
        test_dataset = SRDataset(self.config['data']['train_hr_dir'], self.config['data']['train_lr_dir'],
                                 crop_size=self.config['data']['crop_size'], preload=False)
        if len(test_dataset) > 0:
            lr_sample, hr_sample = test_dataset[0]
            scale_factor = hr_sample.shape[-1] // lr_sample.shape[-1]
            print(f"检测到数据集缩放比例: {scale_factor}x")
        else:
            scale_factor = 4
            print(f"无法从数据集检测缩放比例，使用默认值: {scale_factor}x")

        # 根据任务类型和配置选择相应的模型
        model_type = self.config['model']['type']
        model_variant = self.config['model'].get('model_variant', 'standard')
        
        print(f"创建模型类型: {model_type}, 变体: {model_variant}")
        
        # 根据模型类型创建相应的模型实例
        if model_type == 'FeatureFusionSR':
            self.model = FeatureFusionSR(
                self.config['model'].get('semantic_model_path'), 
                scale=scale_factor
            ).to(self.device)
        elif model_type == 'FeatureFusionSR_NoSemantic':
            self.model = FeatureFusionSR_NoSemantic(
                scale=scale_factor
            ).to(self.device)
        elif model_type == 'FeatureFusionSR_NoCBAM':
            self.model = FeatureFusionSR_NoCBAM(
                self.config['model'].get('semantic_model_path'), 
                scale=scale_factor
            ).to(self.device)
        elif model_type == 'FeatureFusionSR_SingleScale':
            self.model = FeatureFusionSR_SingleScale(
                self.config['model'].get('semantic_model_path'), 
                scale=scale_factor
            ).to(self.device)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
            
        self.criterion_l1 = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['training']['lr'],
                                           weight_decay=self.config['training']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 0.5, 3)

        # 根据设备类型选择合适的scaler
        if self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = torch.amp.GradScaler()

        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)

    def train(self):
        print(f"训练高分辨率图像路径: {self.config['data']['train_hr_dir']}")
        print(f"训练低分辨率图像路径: {self.config['data']['train_lr_dir']}")

        # 不要在训练开始时创建多余的models目录
        # os.makedirs(os.path.join(self.config['training']['output_dir'], "models"), exist_ok=True)

        train_dataset = SRDataset(self.config['data']['train_hr_dir'], self.config['data']['train_lr_dir'],
                                  self.config['data']['crop_size'])
        val_dataset = SRDataset(self.config['data']['val_hr_dir'], self.config['data']['val_lr_dir'],
                                self.config['data']['crop_size'])

        # 打印第一个样本的尺寸检查
        lr_sample, hr_sample = train_dataset[0]
        print(f"数据样本尺寸检查 - LR: {lr_sample.shape}, HR: {hr_sample.shape}")

        train_loader = DataLoader(train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True,
                                  num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=8,
                                pin_memory=True)

        # 创建当前运行的日志目录，并在其中创建models子目录
        current_time = datetime.now().strftime('%Y%m%d-%H%M')
        
        # 根据模型类型或变体添加前缀
        model_type = self.config['model']['type']
        model_variant = self.config['model'].get('model_variant', 'standard')
        
        # 为消融实验添加特定前缀
        if model_type == 'FeatureFusionSR_NoSemantic':
            prefix = 'no_semantic_run_'
        elif model_type == 'FeatureFusionSR_NoCBAM':
            prefix = 'no_cbam_run_'
        elif model_type == 'FeatureFusionSR_SingleScale':
            prefix = 'single_scale_run_'
        else:
            # 标准FFDSR模型保持原样
            prefix = 'run_'
            
        log_dir = os.path.join(self.config['training']['output_dir'], f"{prefix}{current_time}")
        models_dir = os.path.join(log_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        writer = SummaryWriter(log_dir)
        best_val_loss = float('inf')

        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            running_loss = 0.0
            train_tqdm = tqdm(train_loader,
                              desc=f"第 [{epoch + 1}/{self.config['training']['num_epochs']}] 轮 训练中 (lr: {self.optimizer.param_groups[0]['lr']:.6f})")
            for i, (lr_img, hr_img) in enumerate(train_tqdm):
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)

                # 添加形状检查，确认输入数据尺寸正确
                if i == 0 and epoch == 0:
                    print(f"批次尺寸检查 - LR: {lr_img.shape}, HR: {hr_img.shape}")
                    print(
                        f"检查缩放比例: LR高/宽: {lr_img.shape[2]}/{lr_img.shape[3]}, HR高/宽: {hr_img.shape[2]}/{hr_img.shape[3]}")
                    print(
                        f"理论缩放倍数: 高度={hr_img.shape[2] / lr_img.shape[2]}x, 宽度={hr_img.shape[3] / lr_img.shape[3]}x")

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type):
                    # 确保SR输出和HR目标形状完全匹配
                    sr_img = self.model(lr_img)

                    # 为第一批次添加模型输出尺寸检查
                    if i == 0 and epoch == 0:
                        print(f"模型输出形状: {sr_img.shape}，目标形状: {hr_img.shape}")

                    # 如果形状仍然不匹配，使用bicubic插值调整大小
                    if sr_img.shape[2:] != hr_img.shape[2:]:
                        print(f"警告：形状不匹配! SR输出: {sr_img.shape} vs HR目标: {hr_img.shape}")
                        sr_img = F.interpolate(sr_img, size=hr_img.shape[2:], mode='bicubic', align_corners=False)
                        print(f"调整后的SR输出形状: {sr_img.shape}")

                    loss = self.criterion_l1(sr_img, hr_img) / self.config['training']['accumulation_steps']

                self.scaler.scale(loss).backward()
                if (i + 1) % self.config['training']['accumulation_steps'] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                running_loss += loss.item() * self.config['training']['accumulation_steps']
                train_tqdm.set_postfix(loss=loss.item())

            val_loss, val_psnr, val_ssim = self.validate(val_loader)
            writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('PSNR/Val', val_psnr, epoch)
            writer.add_scalar('SSIM/Val', val_ssim, epoch)
            writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            self.scheduler.step(val_loss)
            print(f"Validation Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 直接使用之前创建好的models_dir，不要再创建一次
                torch.save(self.model.state_dict(), os.path.join(models_dir, "best_model.pth"))

    def validate(self, val_loader):
        self.model.eval()
        val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
        val_tqdm = tqdm(val_loader, desc="验证中")
        with torch.no_grad():
            for lr_img, hr_img in val_tqdm:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                sr_img = self.model(lr_img)

                # 确保验证时输入输出形状匹配
                if sr_img.shape != hr_img.shape:
                    sr_img = F.interpolate(sr_img, size=hr_img.shape[2:], mode='bilinear', align_corners=False)

                val_loss += self.criterion_l1(sr_img, hr_img).item()
                val_psnr += self.psnr(sr_img, hr_img).item()
                val_ssim += self.ssim(sr_img, hr_img).item()
                val_tqdm.set_postfix(loss=val_loss, PSNR=val_psnr, SSIM=val_ssim)
        return val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader)