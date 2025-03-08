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
from ..utils.loss import MixedLoss, get_class_weights, create_progressive_weight_scheduler


class SemanticTrainer:
    def __init__(self, config):
        self.config = config
        self.device, device_name = get_device()
        print(f"使用设备: {device_name}")

        # 先定义模型
        self.model = get_semantic_model(self.config['model']['num_classes'],
                                        self.config['model']['pretrained_weights']).to(self.device)

        # 先加载数据集
        print(f"训练图像路径: {self.config['data']['train_img_dir']}")
        print(f"训练标签路径: {self.config['data']['train_label_dir']}")

        self.train_dataset = CityscapesDataset(
            self.config['data']['train_img_dir'],
            self.config['data']['train_label_dir'],
            self.config['data']['crop_size']
        )

        self.val_dataset = CityscapesDataset(
            self.config['data']['val_img_dir'],
            self.config['data']['val_label_dir'],
            self.config['data']['crop_size']
        )

        # 读取权重配置选项
        weight_config = self.config.get('weight', {})
        use_dynamic_weights = weight_config.get('use_dynamic', True)
        use_log_scale = weight_config.get('use_log_scale', False)
        mix_ratio = weight_config.get('mix_ratio', 0.0)
        min_weight = weight_config.get('min_weight', 0.05)
        max_weight = weight_config.get('max_weight', 5.0)
        use_progressive = weight_config.get('use_progressive', False)

        # 计算初始均匀权重和目标权重
        initial_weights = torch.ones(self.config['model']['num_classes'], device=self.device)
        target_weights = get_class_weights(
            self.train_dataset,
            self.config['model']['num_classes'],
            use_predefined=not use_dynamic_weights,
            min_weight=min_weight,
            max_weight=max_weight,
            use_log_scale=use_log_scale,
            mix_ratio=mix_ratio
        ).to(self.device)

        print("目标类别权重:", target_weights)

        if use_progressive:
            print("使用渐进式权重调整策略")
            warmup_epochs = weight_config.get('warmup_epochs', self.config['training']['num_epochs'] // 2)
            self.weight_scheduler = create_progressive_weight_scheduler(
                initial_weights,
                target_weights,
                self.config['training']['num_epochs'],
                warmup_epochs
            )
            self.current_weights = initial_weights
            print(f"初始权重: {self.current_weights}")
            print(f"在 {warmup_epochs} 轮内逐渐过渡到目标权重")
        else:
            print("使用固定权重")
            self.current_weights = target_weights
            self.weight_scheduler = None

        # 使用混合损失函数
        self.criterion = MixedLoss(
            class_weights=self.current_weights,
            focal_weight=weight_config.get('focal_weight', 0.7),
            dice_weight=weight_config.get('dice_weight', 0.3),
            gamma=weight_config.get('gamma', 2.0),
            ignore_index=255
        )

        # 使用SGD+动量代替Adam
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['training'].get('lr', 0.001),
            momentum=0.9,
            weight_decay=self.config['training'].get('weight_decay', 0.0001)
        )

        # 使用余弦退火学习率调度器代替ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 第一次重启的周期
            T_mult=1,  # 每次重启后周期变化倍率
            eta_min=1e-6  # 最小学习率
        )

        self.scaler = torch.amp.GradScaler('cuda')

    def train(self):
        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        current_time = datetime.now().strftime('%Y%m%d-%H%M')
        log_dir = os.path.join(self.config['training']['log_dir'], current_time)
        writer = SummaryWriter(log_dir)
        best_miou = 0.0

        for epoch in range(self.config['training']['num_epochs']):
            # 更新权重（如果使用渐进式调整）
            if self.weight_scheduler is not None:
                self.current_weights = self.weight_scheduler(epoch)
                self.criterion.update_weights(self.current_weights)
                print(f"Epoch {epoch + 1}: 当前使用的权重: {self.current_weights}")

            self.model.train()
            running_loss = 0.0
            train_tqdm = tqdm(train_loader,
                              desc=f"第 [{epoch + 1}/{self.config['training']['num_epochs']}] 轮 训练中 (lr: {self.optimizer.param_groups[0]['lr']:.6f})")
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
            writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # 更新学习率调度器
            self.scheduler.step()
            print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}")

            if val_miou > best_miou:
                best_miou = val_miou
                os.makedirs(self.config['training']['output_dir'], exist_ok=True)
                torch.save(self.model.state_dict(),
                           os.path.join(self.config['training']['output_dir'], "best_semantic_model.pth"))

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