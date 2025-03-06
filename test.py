import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime
import gc
import multiprocessing
from train import SRDataset, FeatureFusionSR

# 获取当前文件所在目录并设置相对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(current_dir, '../../Documents/数据集/CitySpaces/leftImg8bit_trainvaltest/leftImg8bit')
test_hr_dir = os.path.join(base_data_dir, 'test')
test_lr_dir = os.path.join(base_data_dir, 'test_lr_x4')


# 映射分割结果为可视化颜色图
def create_color_map():
    color_map = {
        0: [128, 64, 128],  # 道路
        1: [244, 35, 232],  # 人行道
        2: [70, 70, 70],  # 建筑
        3: [102, 102, 156],  # 墙
        4: [190, 153, 153],  # 栅栏
        5: [153, 153, 153],  # 电线杆
        6: [250, 170, 30],  # 交通灯
        7: [220, 220, 0],  # 交通标志
        8: [107, 142, 35],  # 植被
        9: [152, 251, 152],  # 地形
        10: [70, 130, 180],  # 天空
        11: [220, 20, 60],  # 行人
        12: [255, 0, 0],  # 骑车人
        13: [0, 0, 142],  # 汽车
        14: [0, 0, 70],  # 卡车
        15: [0, 60, 100],  # 公共汽车
        16: [0, 80, 100],  # 火车
        17: [0, 0, 230],  # 摩托车
        18: [119, 11, 32]  # 自行车
    }
    return color_map


def colorize_segmentation(seg_map):
    color_map = create_color_map()
    h, w = seg_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, color in color_map.items():
        mask = (seg_map == label_id)
        colored[mask] = color

    # 转换为PyTorch张量并调整通道顺序 (H,W,3) -> (3,H,W)
    colored = torch.from_numpy(colored.transpose(2, 0, 1)) / 255.0
    return colored


def test_models():
    # 设置输出目录
    output_dir = os.path.join(current_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    lr_seg_dir = os.path.join(output_dir, 'lr_seg')
    sr_img_dir = os.path.join(output_dir, 'sr_img')
    sr_seg_dir = os.path.join(output_dir, 'sr_seg')
    os.makedirs(lr_seg_dir, exist_ok=True)
    os.makedirs(sr_img_dir, exist_ok=True)
    os.makedirs(sr_seg_dir, exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 减少批处理大小以降低内存压力
    batch_size = 2

    # CUDA内存优化
    torch.cuda.empty_cache()
    gc.collect()

    # 加载模型
    semantic_model_path = os.path.join(current_dir, "models/best_semantic_model.pth")
    sr_model_path = os.path.join(current_dir, "runs/run_20250305-163649/models/best_model.pth")

    print(f"加载超分辨率模型: {sr_model_path}")
    sr_model = FeatureFusionSR(semantic_model_path=semantic_model_path)
    sr_model.load_state_dict(torch.load(sr_model_path, map_location=device))
    sr_model = sr_model.to(device)
    sr_model.eval()

    # 加载测试数据集
    print(f"加载测试数据集...")
    test_dataset = SRDataset(hr_dir=test_hr_dir, lr_dir=test_lr_dir, crop_size=None, preload=True)

    # 减少worker数量，降低内存压力
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # 减少worker数量
        pin_memory=True
    )

    # 评估指标
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)

    # 结果存储
    results = []
    avg_psnr = 0.0
    avg_ssim = 0.0
    total_images = 0

    # 创建进度条
    progress_bar = tqdm(test_loader, desc="测试中")

    # 使用try-except捕获CUDA内存错误
    try:
        with torch.no_grad():
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(progress_bar):
                # 逐个图像处理，降低内存压力
                for i in range(lr_imgs.size(0)):
                    lr_img = lr_imgs[i:i + 1].to(device, non_blocking=True)
                    hr_img = hr_imgs[i:i + 1].to(device, non_blocking=True)

                    # 获取LR图像的语义分割
                    lr_semantic = sr_model.semantic_model(lr_img)['out']
                    lr_seg_map = lr_semantic.argmax(dim=1).squeeze().cpu().numpy()

                    # 生成SR图像
                    sr_img = sr_model(lr_img)

                    # 获取SR图像的语义分割
                    sr_semantic = sr_model.semantic_model(sr_img)['out']
                    sr_seg_map = sr_semantic.argmax(dim=1).squeeze().cpu().numpy()

                    # 将分割图转为彩色图像
                    lr_seg_colored = colorize_segmentation(lr_seg_map)
                    sr_seg_colored = colorize_segmentation(sr_seg_map)

                    # 计算指标
                    current_psnr = psnr_metric(sr_img, hr_img).item()
                    current_ssim = ssim_metric(sr_img, hr_img).item()

                    # 更新平均值
                    avg_psnr = (avg_psnr * total_images + current_psnr) / (total_images + 1)
                    avg_ssim = (avg_ssim * total_images + current_ssim) / (total_images + 1)
                    total_images += 1

                    # 记录结果
                    img_name = f"img_{batch_idx}_{i}"
                    results.append({
                        'image': img_name,
                        'psnr': current_psnr,
                        'ssim': current_ssim
                    })

                    # 每10张图像保存一次，减少I/O操作
                    if total_images % 10 == 0:
                        # 保存图像
                        lr_img_cpu = lr_img.squeeze().cpu()
                        sr_img_cpu = sr_img.squeeze().cpu()
                        save_image(lr_img_cpu, os.path.join(output_dir, f'lr_{img_name}.png'))
                        save_image(sr_img_cpu, os.path.join(sr_img_dir, f'{img_name}.png'))
                        save_image(lr_seg_colored, os.path.join(lr_seg_dir, f'{img_name}.png'))
                        save_image(sr_seg_colored, os.path.join(sr_seg_dir, f'{img_name}.png'))

                    # 更新进度条
                    progress_bar.set_postfix({
                        '当前PSNR': f'{current_psnr:.2f}',
                        '当前SSIM': f'{current_ssim:.4f}',
                        '平均PSNR': f'{avg_psnr:.2f}',
                        '平均SSIM': f'{avg_ssim:.4f}'
                    })

                    # 手动释放内存
                    del lr_img, hr_img, sr_img, lr_semantic, sr_semantic

                # 每个批次后清理GPU内存
                torch.cuda.empty_cache()
                if batch_idx % 5 == 0:
                    gc.collect()

    except Exception as e:
        print(f"处理时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 保存已处理的结果
        with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
            f.write(f"平均 PSNR: {avg_psnr:.4f}\n")
            f.write(f"平均 SSIM: {avg_ssim:.4f}\n\n")
            f.write("详细结果:\n")
            for result in results:
                f.write(f"图像: {result['image']}, PSNR: {result['psnr']:.4f}, SSIM: {result['ssim']:.4f}\n")

        print(f"\n测试完成！")
        print(f"平均 PSNR: {avg_psnr:.4f}")
        print(f"平均 SSIM: {avg_ssim:.4f}")
        print(f"详细结果已保存到 {os.path.join(output_dir, 'results.txt')}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    test_models()