import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models import resnet18
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# 定义 SEBlock 类
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 定义 EDSR 类
class EDSR(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, num_blocks=8, upscale_factor=4):
        super(EDSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_blocks)]
        )
        self.conv2 = nn.Conv2d(64, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.body(x)
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        return x

# 定义 FeatureFusionSR 类
class FeatureFusionSR(nn.Module):
    def __init__(self):
        super(FeatureFusionSR, self).__init__()
        self.semantic_model = deeplabv3_resnet101(weights='DEFAULT').eval()
        self.embedding = nn.Embedding(19, 64)

        self.resnet = resnet18(weights='DEFAULT')
        self.resnet_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.resnet_bn1 = self.resnet.bn1
        self.resnet_relu = self.resnet.relu
        self.resnet_layer1 = self.resnet.layer1

        self.se_block = SEBlock(128)
        self.fusion_conv = nn.Conv2d(128, 256, kernel_size=1)

        self.sr_net = EDSR()

    def forward(self, lr_img):
        with torch.no_grad():
            semantic_output = self.semantic_model(lr_img)['out']
        semantic_labels = semantic_output.argmax(dim=1)
        semantic_feature = self.embedding(semantic_labels).permute(0, 3, 1, 2)

        x = self.resnet_conv1(lr_img)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        low_level_feature = self.resnet_layer1(x)

        target_h, target_w = low_level_feature.shape[2:]
        semantic_feature = F.interpolate(
            semantic_feature, size=(target_h, target_w), mode='bilinear', align_corners=False
        )

        fused_feature = torch.cat([semantic_feature, low_level_feature], dim=1)
        fused_feature = self.se_block(fused_feature)
        fused_feature = self.fusion_conv(fused_feature)

        sr_img = self.sr_net(fused_feature)
        return sr_img

# 测试函数
def test_super_resolution(model_path, test_dir, output_dir):
    # 动态选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("No GPU available, using device: CPU")

    # 加载模型
    model = FeatureFusionSR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 数据转换
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # 获取测试图片列表
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # 处理每张图片
    with torch.no_grad():
        for img_name in tqdm(test_images, desc="Processing images"):
            # 读取低分辨率图片
            img_path = os.path.join(test_dir, img_name)
            lr_img = Image.open(img_path).convert('RGB')
            lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

            # 生成超分辨率图片
            sr_tensor = model(lr_tensor)
            sr_tensor = sr_tensor.squeeze(0).cpu()

            # 保存结果
            sr_img = to_pil(sr_tensor)
            output_path = os.path.join(output_dir, f"sr_{img_name}")
            sr_img.save(output_path)

    print(f"超清重建完成！结果已保存至 {output_dir}")

# 主函数
if __name__ == "__main__":
    # 配置路径
    model_path = "models/final_model.pth"  # 训练好的模型路径
    test_dir = "test_images"              # 测试图片文件夹
    output_dir = "output_images"          # 输出文件夹

    # 运行测试
    test_super_resolution(model_path, test_dir, output_dir)