import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
from ..datasets.semantic_dataset import CityscapesDataset
from ..models.semantic_model import get_semantic_model
from ..utils.metrics import calculate_miou
from ..utils.common import get_device

class SemanticTrainer:
    def __init__(self, config):
        self.config = config
        self.device, device_name = get_device()
        print(f"使用设备: {device_name}")
        self.model = get_semantic_model(self.config['model']['num_classes'], self.config['model']['pretrained_weights']).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['training']['lr'], weight_decay=self.config['training']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.scaler = torch.amp.GradScaler('cuda')

    def train(self):
        print(f"训练图像路径: {self.config['data']['train_img_dir']}")
        print(f"训练标签路径: {self.config['data']['train_label_dir']}")
        train_dataset = CityscapesDataset(self.config['data']['train_img_dir'], self.config['data']['train_label_dir'], self.config['data']['crop_size'])
        val_dataset = CityscapesDataset(self.config['data']['val_img_dir'], self.config['data']['val_label_dir'], self.config['data']['crop_size'])
        train_loader = DataLoader(train_dataset, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

        current_time = datetime.now().strftime('%Y%m%d-%H%M')
        log_dir = os.path.join(self.config['training']['log_dir'], current_time)
        writer = SummaryWriter(log_dir)
        best_miou = 0.0

        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            running_loss = 0.0
            train_tqdm = tqdm(train_loader, desc=f"第 [{epoch+1}/{self.config['training']['num_epochs']}] 轮 训练中")
            for i, (imgs, labels, _) in enumerate(train_tqdm):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = self.model(imgs)['out']
                    loss = self.criterion(outputs, labels) / self.config['training']['accumulation_steps']
                self.scaler.scale(loss).backward()
                if (i + 1) % self.config['training']['accumulation_steps'] == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                running_loss += loss.item() * self.config['training']['accumulation_steps']
                train_tqdm.set_postfix(loss=loss.item())

            val_loss, val_miou = self.validate(val_loader)
            writer.add_scalar('Loss/Train', running_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('mIoU/Val', val_miou, epoch)
            self.scheduler.step(val_loss)
            print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}")

            if val_miou > best_miou:
                best_miou = val_miou
                os.makedirs(self.config['training']['output_dir'], exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.config['training']['output_dir'], "best_semantic_model.pth"))

    def validate(self, val_loader):
        self.model.eval()
        val_loss, val_miou = 0.0, 0.0
        val_tqdm = tqdm(val_loader, desc="验证中")
        with torch.no_grad():
            for imgs, labels, _ in val_tqdm:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)['out']
                val_loss += self.criterion(outputs, labels).item()
                val_miou += calculate_miou(outputs, labels)
                val_tqdm.set_postfix(loss=val_loss, mIoU=val_miou)
        return val_loss / len(val_loader), val_miou / len(val_loader)