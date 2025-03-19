import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import torch.nn.functional as F
from ..datasets.sr_dataset import SRDataset
from ..models.srcnn_model import SRCNN
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from ..utils.common import get_device


class SRCNNTester:
    def __init__(self, config):
        self.config = config
        self.device, device_name = get_device()
        print(f"使用设备: {device_name}")
        
        # 确定缩放因子
        test_dataset = SRDataset(
            self.config['data']['test_hr_dir'], 
            self.config['data']['test_lr_dir'],
            preload=False
        )
        if len(test_dataset) > 0:
            lr_sample, hr_sample = test_dataset[0]
            scale_factor = hr_sample.shape[-1] // lr_sample.shape[-1]
            print(f"检测到数据集缩放比例: {scale_factor}x")
        else:
            scale_factor = 4
            print(f"无法从数据集检测缩放比例，使用默认值: {scale_factor}x")
        
        # 初始化模型
        self.model = SRCNN(num_channels=3, scale=scale_factor).to(self.device)
        
        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load(self.config['model']['model_path'], map_location=self.device))
            print(f"模型权重加载成功: {self.config['model']['model_path']}")
        except Exception as e:
            print(f"模型权重加载失败: {e}")
            exit(1)
            
        self.model.eval()
        
        # 评估指标
        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)

    def test(self):
        # 创建数据集和数据加载器
        dataset = SRDataset(
            self.config['data']['test_hr_dir'], 
            self.config['data']['test_lr_dir'], 
            preload=True
        )
        loader = DataLoader(
            dataset, 
            batch_size=self.config['data']['batch_size'], 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        # 创建输出目录
        os.makedirs(self.config['output']['dir'], exist_ok=True)
        results_file = os.path.join(self.config['output']['dir'], 'srcnn_results.txt')

        avg_psnr, avg_ssim, total_images = 0.0, 0.0, 0
        all_results = []
        
        with torch.no_grad():
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(tqdm(loader, desc="SRCNN测试中")):
                for i in range(lr_imgs.size(0)):
                    lr_img = lr_imgs[i:i+1].to(self.device)
                    hr_img = hr_imgs[i:i+1].to(self.device)
                    
                    # 生成SR图像
                    sr_img = self.model(lr_img)
                    
                    # 确保尺寸匹配
                    if sr_img.shape != hr_img.shape:
                        sr_img = F.interpolate(sr_img, size=hr_img.shape[2:], mode='bicubic', align_corners=False)
                    
                    # 计算指标
                    psnr_val = self.psnr(sr_img, hr_img).item()
                    ssim_val = self.ssim(sr_img, hr_img).item()
                    
                    # 累积平均值
                    old_avg_psnr, old_avg_ssim = avg_psnr, avg_ssim
                    avg_psnr = (avg_psnr * total_images + psnr_val) / (total_images + 1)
                    avg_ssim = (avg_ssim * total_images + ssim_val) / (total_images + 1)
                    total_images += 1
                    
                    # 记录当前图像结果
                    all_results.append(f"图像 {total_images}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")
                    
                    # 每10张图像保存一次结果图
                    if total_images % 10 == 0:
                        # 保存LR图像
                        save_image(lr_img, os.path.join(
                            self.config['output']['dir'], 
                            f"srcnn_lr_{batch_idx}_{i}.png"
                        ))
                        # 保存SR结果
                        save_image(sr_img, os.path.join(
                            self.config['output']['dir'], 
                            f"srcnn_sr_{batch_idx}_{i}.png"
                        ))
                        # 保存HR原图
                        save_image(hr_img, os.path.join(
                            self.config['output']['dir'], 
                            f"srcnn_hr_{batch_idx}_{i}.png"
                        ))
        
        # 输出和保存结果
        print(f"SRCNN 平均 PSNR: {avg_psnr:.4f}, 平均 SSIM: {avg_ssim:.4f}, 总图像数: {total_images}")
        
        # 将结果写入文件
        with open(results_file, 'w') as f:
            f.write(f"SRCNN 测试结果\n")
            f.write(f"总图像数: {total_images}\n")
            f.write(f"平均 PSNR: {avg_psnr:.4f}\n")
            f.write(f"平均 SSIM: {avg_ssim:.4f}\n")
            f.write("\n详细结果:\n")
            for result in all_results:
                f.write(f"{result}\n")
                
        print(f"测试结果已保存到: {results_file}")
