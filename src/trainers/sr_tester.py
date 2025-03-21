import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
from ..datasets.sr_dataset import SRDataset
from ..models.sr_model import (
    FeatureFusionSR,
    FeatureFusionSR_NoSemantic,
    FeatureFusionSR_NoCBAM,
    FeatureFusionSR_SingleScale
)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from ..utils.common import get_device

class SRTester:
    def __init__(self, config):
        self.config = config
        self.device, device_name = get_device()
        print(f"使用设备: {device_name}")
        
        # 根据模型类型创建相应的模型实例
        model_type = self.config['model']['type']
        model_variant = self.config['model'].get('model_variant', 'standard')
        
        print(f"创建模型类型: {model_type}, 变体: {model_variant}")
        
        if model_type == 'FeatureFusionSR':
            self.model = FeatureFusionSR(
                self.config['model'].get('semantic_model_path')
            ).to(self.device)
        elif model_type == 'FeatureFusionSR_NoSemantic':
            self.model = FeatureFusionSR_NoSemantic().to(self.device)
        elif model_type == 'FeatureFusionSR_NoCBAM':
            self.model = FeatureFusionSR_NoCBAM(
                self.config['model'].get('semantic_model_path')
            ).to(self.device)
        elif model_type == 'FeatureFusionSR_SingleScale':
            self.model = FeatureFusionSR_SingleScale(
                self.config['model'].get('semantic_model_path')
            ).to(self.device)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
            
        # 加载模型权重
        model_path = self.config['model']['model_path']
        print(f"加载模型权重: {model_path}")
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("模型权重加载成功")
        except Exception as e:
            print(f"加载模型权重时出错: {e}")
            exit(1)
            
        self.model.eval()
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)

    def test(self):
        dataset = SRDataset(self.config['data']['test_hr_dir'], self.config['data']['test_lr_dir'], preload=True)
        loader = DataLoader(dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        os.makedirs(self.config['output']['dir'], exist_ok=True)

        avg_psnr, avg_ssim, total_images = 0.0, 0.0, 0
        with torch.no_grad():
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(tqdm(loader, desc="测试中")):
                for i in range(lr_imgs.size(0)):
                    lr_img = lr_imgs[i:i+1].to(self.device)
                    hr_img = hr_imgs[i:i+1].to(self.device)
                    sr_img = self.model(lr_img)
                    psnr_val = self.psnr(sr_img, hr_img).item()
                    ssim_val = self.ssim(sr_img, hr_img).item()
                    avg_psnr = (avg_psnr * total_images + psnr_val) / (total_images + 1)
                    avg_ssim = (avg_ssim * total_images + ssim_val) / (total_images + 1)
                    total_images += 1
                    if total_images % 10 == 0:
                        save_image(sr_img, os.path.join(self.config['output']['dir'], f"sr_img_{batch_idx}_{i}.png"))
        print(f"平均 PSNR: {avg_psnr:.4f}, 平均 SSIM: {avg_ssim:.4f}")