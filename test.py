import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


# 定义 ChannelAttention 类
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return self.sigmoid(avg_out + max_out).view(b, c, 1, 1) * x


# 定义 SpatialAttention 类
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x


# 定义 CBAM 类
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# 定义 ResBlock 类
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return residual + self.scale * out


# 定义 EnhancedEDSR 类
class EnhancedEDSR(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, num_blocks=32):
        super(EnhancedEDSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(64) for _ in range(num_blocks)]
        )
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_up1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.att1 = CBAM(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.att2 = CBAM(64)
        self.fusion = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.body(x)
        x2 = self.conv2(x)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = self.conv_up1(x2)
        x2 = self.att1(x2)
        x3 = self.conv3(x2)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = self.conv_up2(x3)
        x3 = self.att2(x3)
        x_fused = self.fusion(
            torch.cat([F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False), x3], dim=1))
        x = self.conv_out(x_fused)
        return torch.sigmoid(x)


# 定义 FeatureFusionSR 类
class FeatureFusionSR(nn.Module):
    def __init__(self):
        super(FeatureFusionSR, self).__init__()
        self.semantic_model = deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1').eval()
        self.embedding = nn.Embedding(91, 64)
        self.resnet = models.resnet50(weights='DEFAULT')
        self.resnet_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.resnet_bn1 = self.resnet.bn1
        self.resnet_relu = self.resnet.relu
        self.resnet_maxpool = nn.Identity()
        self.resnet_layer1 = self.resnet.layer1
        self.resnet_layer2 = self.resnet.layer2
        self.resnet_layer3 = self.resnet.layer3
        self.fusion_conv = nn.Conv2d(1856, 256, kernel_size=1)
        self.cbam = CBAM(256)
        self.sr_net = EnhancedEDSR()

        for param in self.semantic_model.backbone.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet_layer1.parameters():
            param.requires_grad = False
        for param in self.resnet_layer2.parameters():
            param.requires_grad = False

    def forward_semantic(self, lr_img):
        with torch.no_grad():
            semantic_output = self.semantic_model(lr_img)['out']
        semantic_labels = semantic_output.argmax(dim=1)
        semantic_feature = self.embedding(semantic_labels).permute(0, 3, 1, 2)
        return semantic_feature

    def forward_low_level(self, lr_img):
        x = self.resnet_conv1(lr_img)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)
        layer1_out = self.resnet_layer1(x)
        layer2_out = self.resnet_layer2(layer1_out)
        layer3_out = self.resnet_layer3(layer2_out)
        layer2_out = F.interpolate(layer2_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False)
        layer3_out = F.interpolate(layer3_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False)
        low_level_feature = torch.cat([layer1_out, layer2_out, layer3_out], dim=1)
        return low_level_feature

    def forward_fusion_and_sr(self, semantic_feature, low_level_feature):
        target_h, target_w = low_level_feature.shape[2:]
        semantic_feature = F.interpolate(semantic_feature, size=(target_h, target_w), mode='bilinear',
                                         align_corners=False)
        fused_feature = torch.cat([semantic_feature, low_level_feature], dim=1)
        fused_feature = self.fusion_conv(fused_feature)
        fused_feature = self.cbam(fused_feature)
        sr_img = self.sr_net(fused_feature)
        return sr_img

    def forward(self, lr_img):
        semantic_feature = self.forward_semantic(lr_img)
        low_level_feature = self.forward_low_level(lr_img)
        sr_img = self.forward_fusion_and_sr(semantic_feature, low_level_feature)
        return sr_img


# 测试函数
def test_super_resolution(model_path, test_dir, output_dir):
    # 动态选择设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用设备: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用设备: MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("没有可用的GPU，使用设备: CPU")

    # 加载模型
    model = FeatureFusionSR().to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        return

    model.eval()

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 数据转换
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # 获取测试图片列表
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not test_images:
        print(f"测试文件夹 {test_dir} 中没有找到图片！")
        return

    # 处理每张图片
    with torch.no_grad():
        for img_name in tqdm(test_images, desc="处理图片"):
            # 读取低分辨率图片
            img_path = os.path.join(test_dir, img_name)
            try:
                lr_img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"无法打开图片 {img_path}: {e}")
                continue

            lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

            # 生成超分辨率图片
            try:
                sr_tensor = model(lr_tensor)
                sr_tensor = torch.clamp(sr_tensor, 0, 1)  # 确保输出在 [0, 1] 范围内
                sr_tensor = sr_tensor.squeeze(0).cpu()
            except Exception as e:
                print(f"处理图片 {img_name} 时出错: {e}")
                continue

            # 保存结果
            sr_img = to_pil(sr_tensor)
            output_path = os.path.join(output_dir, f"sr_{img_name}")
            sr_img.save(output_path)

    print(f"超分辨率重建完成！结果已保存至 {output_dir}")


# 主函数
if __name__ == "__main__":
    # 配置路径
    model_path = "models/best_model.pth"  # 训练好的模型路径
    test_dir = "test_images"  # 测试图片文件夹
    output_dir = "output_images"  # 输出文件夹

    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在！")
    elif not os.path.exists(test_dir):
        print(f"测试图片文件夹 {test_dir} 不存在！")
    else:
        # 运行测试
        test_super_resolution(model_path, test_dir, output_dir)