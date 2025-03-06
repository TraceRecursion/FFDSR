import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
from ..datasets.sr_dataset import SRDataset
from ..models.sr_model import FeatureFusionSR
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class SRTrainer:
    def __init__(self, config):
        self.config = config
        self.device, device_name = torch.cuda.is_available() and (torch.device("cuda"), "CUDA") or (torch.device("cpu"), "CPU")
        print(f"使用设备: {device_name}")
        self.model = FeatureFusionSR(self.config['model']['semantic_model_path']).to(self.device)
        self.criterion_l1 = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['training']['lr'], weight_decay=self.config['training']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', 0.5, 3)
        self.scaler = torch.amp.GradScaler('cuda')
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)

    def train(self):
        print(f"训练图像路径: {self.config['data']['train_img_dir']}")
        print(f"训练标签路径: {self.config['data']['train_label_dir']}")
        train_dataset = SRDataset(self.config['data']['train_hr_dir'], self.config['data']['train_lr_dir'], self.config['data']['crop_size'])
        val_dataset = SRDataset(self.config['data']['val_hr_dir'], self.config['data']['val_lr_dir'], self.config['data']['crop_size'])
        train_loader = DataLoader(train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

        current_time = datetime.now().strftime('%Y%m%d-%H%M')
        log_dir = os.path.join(self.config['training']['output_dir'], current_time)
        writer = SummaryWriter(log_dir)
        best_val_loss = float('inf')

        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            running_loss = 0.0
            train_tqdm = tqdm(train_loader, desc=f"第 [{epoch+1}/{self.config['training']['num_epochs']}] 轮 训练中 (lr: {self.optimizer.param_groups[0]['lr']:.6f})")
            for i, (lr_img, hr_img) in enumerate(train_tqdm):
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    sr_img = self.model(lr_img)
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
                os.makedirs(self.config['training']['output_dir'], exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.config['training']['output_dir'], "models/best_model.pth"))

    def validate(self, val_loader):
        self.model.eval()
        val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
        val_tqdm = tqdm(val_loader, desc="验证中")
        with torch.no_grad():
            for lr_img, hr_img in val_tqdm:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                sr_img = self.model(lr_img)
                val_loss += self.criterion_l1(sr_img, hr_img).item()
                val_psnr += self.psnr(sr_img, hr_img).item()
                val_ssim += self.ssim(sr_img, hr_img).item()
                val_tqdm.set_postfix(loss=val_loss, PSNR=val_psnr, SSIM=val_ssim)
        return val_loss / len(val_loader), val_psnr / len(val_loader), val_ssim / len(val_loader)