import torch
from torch.utils.data import DataLoader
from ..datasets.semantic_dataset import CityscapesDataset
from ..models.semantic_model import get_semantic_model
from ..utils.visualize import visualize_segmentation
from ..utils.common import get_device
import os
import gc
from PIL import Image
from tqdm import tqdm
import time


class SemanticTester:
    def __init__(self, config):
        self.config = config
        self.device, device_name = get_device()
        print(f"使用设备: {device_name}")
        
        # 获取预训练权重路径 - 兼容model_path和pretrained_weights两种配置
        if 'model_path' in self.config['model']:
            checkpoint_path = self.config['model']['model_path']
        elif 'pretrained_weights' in self.config['model']:
            checkpoint_path = self.config['model']['pretrained_weights']
        else:
            print("错误: 配置文件中缺少模型权重路径，请检查配置文件")
            checkpoint_path = None
            
        # 初始化模型
        self.model = get_semantic_model(
            self.config['model']['num_classes'],
            checkpoint_path
        ).to(self.device)
        
        print("模型初始化完成")
        self.model.eval()

    def test(self):
        dataset = CityscapesDataset(
            self.config['data']['test_img_dir'],
            self.config['data']['test_label_dir'] if 'test_label_dir' in self.config['data'] else None,
            self.config['data']['crop_size'],
            test_mode=True
        )

        print(f"测试数据集大小: {len(dataset)}张图像")
        os.makedirs(self.config['output']['dir'], exist_ok=True)

        progress_bar = tqdm(range(len(dataset)), desc="语义分割测试")
        start_time = time.time()
        processed_images = 0
        
        try:
            for idx in progress_bar:
                try:
                    if dataset.label_dir:
                        img_tensor, label_tensor, img_name = dataset[idx]
                    else:
                        img_tensor, img_name = dataset[idx]
                        
                    img_path = dataset.images[idx]
                    
                    # 简化描述信息，只显示当前处理的图像名称
                    progress_bar.set_description(f"处理: {img_name}")

                    # 确保原始图像不会太大
                    orig_img = Image.open(img_path).convert('RGB')
                    
                    # 如果图像太大，调整大小以减少内存使用
                    max_dim = 1024
                    if max(orig_img.width, orig_img.height) > max_dim:
                        ratio = max_dim / max(orig_img.width, orig_img.height)
                        new_w = int(orig_img.width * ratio)
                        new_h = int(orig_img.height * ratio)
                        orig_img = orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(img_tensor)['out']
                        pred = outputs.argmax(dim=1).cpu().numpy()[0]
                        
                        # 清理不需要的张量
                        del outputs
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                        # 确保结果目录存在
                        output_dir = self.config['output']['dir']
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 可视化
                        result_img = visualize_segmentation(pred, orig_img)
                        save_path = os.path.join(output_dir, f"pred_{img_name}")
                        result_img.save(save_path, quality=90)
                    
                    processed_images += 1
                    
                    # 释放内存
                    del img_tensor, pred
                    if 'label_tensor' in locals():
                        del label_tensor
                    
                    # 每5张图片清空缓存，防止内存泄漏
                    if idx % 5 == 0:
                        if self.device.type == 'mps':
                            torch.mps.empty_cache()
                        elif torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    print(f"\n处理图像 '{img_name}' 时出错: {str(e)}")
                    continue
        
        except KeyboardInterrupt:
            print("\n测试过程被用户中断")
        except Exception as e:
            print(f"\n处理过程中发生错误: {str(e)}")
        
        finally:
            # 计算总统计信息
            total_time = time.time() - start_time
            avg_time_per_img = total_time / processed_images if processed_images > 0 else 0
            print(f"\n语义分割测试完成! 已处理 {processed_images}/{len(dataset)} 张图像")
            print(f"总用时: {int(total_time//60)}分{int(total_time%60)}秒, 平均每张图像 {avg_time_per_img:.2f} 秒")
            print(f"结果保存至: {self.config['output']['dir']}")