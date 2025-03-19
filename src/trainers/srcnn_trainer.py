import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
from ..datasets.sr_dataset import SRDataset
from ..models.srcnn_model import SRCNN
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from ..utils.common import get_device


class SRCNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device, device_name = get_device()
        print(f"使用设备: {device_name}")

        # 确定模型缩放比例
        test_dataset = SRDataset(
            self.config['data']['train_hr_dir'], 
            self.config['data']['train_lr_dir'],
            crop_size=self.config['data']['crop_size'], 
            preload=False
        )
        if len(test_dataset) > 0:
            lr_sample, hr_sample = test_dataset[0]
            scale_factor = hr_sample.shape[-1] // lr_sample.shape[-1]
            print(f"检测到数据集缩放比例: {scale_factor}x")
        else:
            scale_factor = 4
            print(f"无法从数据集检测缩放比例，使用默认值: {scale_factor}x")

        # 初始化SRCNN模型
        self.model = SRCNN(num_channels=3, scale=scale_factor).to(self.device)
        
        # 损失函数：MSE损失，SRCNN原始论文中使用的损失函数
        self.criterion = nn.MSELoss()
        
        # 优化器：使用SGD，学习率设定为0.0001，与经典SRCNN实现一致
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.config['training'].get('lr', 0.0001),
            momentum=0.9,
            weight_decay=self.config['training'].get('weight_decay', 0.0001)
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.5
        )

        # 评估指标
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)

    def train(self):
        print(f"训练高分辨率图像路径: {self.config['data']['train_hr_dir']}")
        print(f"训练低分辨率图像路径: {self.config['data']['train_lr_dir']}")

        # 创建数据集
        train_dataset = SRDataset(
            self.config['data']['train_hr_dir'], 
            self.config['data']['train_lr_dir'],
            self.config['data']['crop_size']
        )
        val_dataset = SRDataset(
            self.config['data']['val_hr_dir'], 
            self.config['data']['val_lr_dir'],
            self.config['data']['crop_size']
        )

        # 打印样本尺寸检查
        lr_sample, hr_sample = train_dataset[0]
        print(f"数据样本尺寸检查 - LR: {lr_sample.shape}, HR: {hr_sample.shape}")

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['data']['batch_size'], 
            shuffle=True,
            num_workers=8, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['data']['batch_size'], 
            shuffle=False, 
            num_workers=8,
            pin_memory=True
        )

        # 创建日志目录
        current_time = datetime.now().strftime('%Y%m%d-%H%M')
        log_dir = os.path.join(self.config['training']['output_dir'], f"srcnn_run_{current_time}")
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

                # 批次尺寸检查，仅在第一个批次
                if i == 0 and epoch == 0:
                    print(f"批次尺寸检查 - LR: {lr_img.shape}, HR: {hr_img.shape}")
                    print(f"缩放比例: 高度={hr_img.shape[2]/lr_img.shape[2]}x, 宽度={hr_img.shape[3]/lr_img.shape[3]}x")

                # 前向传播
                self.optimizer.zero_grad()
                sr_img = self.model(lr_img)
                
                # 确保形状匹配
                if sr_img.shape != hr_img.shape:
                    sr_img = F.interpolate(sr_img, size=hr_img.shape[2:], mode='bicubic', align_corners=False)
                    if i == 0 and epoch == 0:
                        print(f"调整后的形状: {sr_img.shape}")
                
                # 计算损失
                loss = self.criterion(sr_img, hr_img)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                train_tqdm.set_postfix(loss=loss.item())
            
            # 验证
            val_loss, val_psnr, val_ssim = self.validate(val_loader)
            
            # 记录指标
            writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('PSNR/Val', val_psnr, epoch)
            writer.add_scalar('SSIM/Val', val_ssim, epoch)
            writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            print(f"Validation Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(models_dir, "best_srcnn_model.pth"))
                print(f"保存最佳模型，验证损失为: {val_loss:.4f}")

    def validate(self, val_loader):
        self.model.eval()
        val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
        val_tqdm = tqdm(val_loader, desc="验证中")
        
        with torch.no_grad():
            for lr_img, hr_img in val_tqdm:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                sr_img = self.model(lr_img)
                
                # 确保形状匹配
                if sr_img.shape != hr_img.shape:
                    sr_img = F.interpolate(sr_img, size=hr_img.shape[2:], mode='bicubic', align_corners=False)
                
                # 计算损失和指标
                val_loss += self.criterion(sr_img, hr_img).item()
                val_psnr += self.psnr(sr_img, hr_img).item()
                val_ssim += self.ssim(sr_img, hr_img).item()
                
                val_tqdm.set_postfix(loss=val_loss/(val_tqdm.n+1), PSNR=val_psnr/(val_tqdm.n+1), SSIM=val_ssim/(val_tqdm.n+1))
                
        return val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader)
