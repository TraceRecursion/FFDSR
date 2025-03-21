import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .semantic_model import get_semantic_model
import os


class CBAM(nn.Module):
    # 从原代码中提取并简化
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = self.channel_att(x) * x
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial = self.spatial_att(torch.cat([avg_pool, max_pool], dim=1)) * x
        return spatial


class EnhancedEDSR(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, num_blocks=32, scale=4):
        super(EnhancedEDSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) for _ in range(num_blocks)])

        # 完全重写上采样部分，确保确实产生4倍放大
        self.scale = scale
        if scale == 4:
            # 四倍上采样 - 明确的实现
            self.upscale = nn.Sequential(
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )
        elif scale == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            )
        else:
            # 对于其他比例使用直接插值
            self.upscale = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.cbam = CBAM(64)
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)

        print(f"初始化EnhancedEDSR，目标缩放倍数: {scale}x")

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.body(x)
        x = x + residual

        # 上采样处理
        x = self.upscale(x)

        # 如果使用的是非2或4倍上采样，需要额外的插值
        if self.scale not in [2, 4]:
            x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # 打印形状用于调试
        # print(f"上采样后形状: {x.shape}")

        x = self.cbam(x)
        return torch.sigmoid(self.conv_out(x))


class FeatureFusionSR(nn.Module):
    def __init__(self, semantic_model_path=None, scale=4):
        super(FeatureFusionSR, self).__init__()
        # 使用更健壮的语义模型加载方式
        self.semantic_model = get_semantic_model(num_classes=19, pretrained_weights=semantic_model_path)

        # 冻结语义模型参数
        self.semantic_model.eval()
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.embedding = nn.Embedding(19, 64)
        self.resnet = models.resnet50(weights='DEFAULT')
        self.fusion_conv = nn.Conv2d(1856, 256, 1)
        self.cbam = CBAM(256)
        self.sr_net = EnhancedEDSR(scale=scale)
        print(f"初始化SR模型, 缩放比例: {scale}x")

    def forward(self, lr_img):
        # 记录输入的形状以便调试
        input_shape = lr_img.shape
        # print(f"SR模型输入形状: {input_shape}")

        with torch.no_grad():
            semantic_out = self.semantic_model(lr_img)['out']
        semantic_labels = semantic_out.argmax(dim=1)
        semantic_feature = self.embedding(semantic_labels).permute(0, 3, 1, 2)

        x = self.resnet.conv1(lr_img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        layer1_out = self.resnet.layer1(x)
        layer2_out = self.resnet.layer2(layer1_out)
        layer3_out = self.resnet.layer3(layer2_out)
        low_level_feature = torch.cat([
            layer1_out,
            F.interpolate(layer2_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(layer3_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)

        fused_feature = self.fusion_conv(torch.cat([
            F.interpolate(semantic_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False),
            low_level_feature
        ], dim=1))

        sr_out = self.sr_net(self.cbam(fused_feature))
        # print(f"SR模型输出形状: {sr_out.shape}, 预期形状: {input_shape[0]}, {input_shape[1]}, {input_shape[2]*4}, {input_shape[3]*4}")

        # 确保输出尺寸是输入的4倍
        if sr_out.shape[2] != input_shape[2] * 4 or sr_out.shape[3] != input_shape[3] * 4:
            sr_out = F.interpolate(sr_out, size=(input_shape[2] * 4, input_shape[3] * 4),
                                   mode='bicubic', align_corners=False)
            # print(f"调整后的SR输出形状: {sr_out.shape}")

        return sr_out


# 消融实验1：无语义特征的变体
class FeatureFusionSR_NoSemantic(nn.Module):
    def __init__(self, semantic_model_path=None, scale=4):
        super(FeatureFusionSR_NoSemantic, self).__init__()
        # 不使用语义模型，直接去掉语义分支
        
        self.resnet = models.resnet50(weights='DEFAULT')
        self.fusion_conv = nn.Conv2d(1792, 256, 1)  # 减小输入通道数，不再包含语义特征
        self.cbam = CBAM(256)
        self.sr_net = EnhancedEDSR(scale=scale)
        print(f"初始化SR模型 (无语义特征), 缩放比例: {scale}x")

    def forward(self, lr_img):
        # 记录输入的形状以便调试
        input_shape = lr_img.shape
        
        # 只提取ResNet特征，不使用语义特征
        x = self.resnet.conv1(lr_img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        layer1_out = self.resnet.layer1(x)
        layer2_out = self.resnet.layer2(layer1_out)
        layer3_out = self.resnet.layer3(layer2_out)
        low_level_feature = torch.cat([
            layer1_out,
            F.interpolate(layer2_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(layer3_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)

        # 没有语义特征融合，直接使用ResNet特征
        fused_feature = self.fusion_conv(low_level_feature)
        
        sr_out = self.sr_net(self.cbam(fused_feature))

        # 确保输出尺寸是输入的4倍
        if sr_out.shape[2] != input_shape[2] * 4 or sr_out.shape[3] != input_shape[3] * 4:
            sr_out = F.interpolate(sr_out, size=(input_shape[2] * 4, input_shape[3] * 4),
                                   mode='bicubic', align_corners=False)

        return sr_out


# 消融实验2：无CBAM的变体
class FeatureFusionSR_NoCBAM(nn.Module):
    def __init__(self, semantic_model_path=None, scale=4):
        super(FeatureFusionSR_NoCBAM, self).__init__()
        # 使用语义模型但移除CBAM注意力机制
        self.semantic_model = get_semantic_model(num_classes=19, pretrained_weights=semantic_model_path)

        # 冻结语义模型参数
        self.semantic_model.eval()
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.embedding = nn.Embedding(19, 64)
        self.resnet = models.resnet50(weights='DEFAULT')
        self.fusion_conv = nn.Conv2d(1856, 256, 1)
        # 移除CBAM
        self.sr_net = EnhancedEDSR(scale=scale)
        print(f"初始化SR模型 (无CBAM), 缩放比例: {scale}x")

    def forward(self, lr_img):
        # 记录输入的形状以便调试
        input_shape = lr_img.shape
        
        with torch.no_grad():
            semantic_out = self.semantic_model(lr_img)['out']
        semantic_labels = semantic_out.argmax(dim=1)
        semantic_feature = self.embedding(semantic_labels).permute(0, 3, 1, 2)

        x = self.resnet.conv1(lr_img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        layer1_out = self.resnet.layer1(x)
        layer2_out = self.resnet.layer2(layer1_out)
        layer3_out = self.resnet.layer3(layer2_out)
        low_level_feature = torch.cat([
            layer1_out,
            F.interpolate(layer2_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(layer3_out, size=layer1_out.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)

        fused_feature = self.fusion_conv(torch.cat([
            F.interpolate(semantic_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False),
            low_level_feature
        ], dim=1))
        
        # 直接将融合特征传递给SR网络，不使用CBAM
        sr_out = self.sr_net(fused_feature)

        # 确保输出尺寸是输入的4倍
        if sr_out.shape[2] != input_shape[2] * 4 or sr_out.shape[3] != input_shape[3] * 4:
            sr_out = F.interpolate(sr_out, size=(input_shape[2] * 4, input_shape[3] * 4),
                                   mode='bicubic', align_corners=False)

        return sr_out


# 消融实验3：无多尺度特征的变体 (仅使用Layer3)
class FeatureFusionSR_SingleScale(nn.Module):
    def __init__(self, semantic_model_path=None, scale=4):
        super(FeatureFusionSR_SingleScale, self).__init__()
        # 使用语义模型，但仅使用ResNet的Layer3特征
        self.semantic_model = get_semantic_model(num_classes=19, pretrained_weights=semantic_model_path)

        # 冻结语义模型参数
        self.semantic_model.eval()
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.embedding = nn.Embedding(19, 64)
        self.resnet = models.resnet50(weights='DEFAULT')
        # 只使用layer3特征，减少融合通道
        self.fusion_conv = nn.Conv2d(1088, 256, 1)  # 1024 (layer3) + 64 (semantic)
        self.cbam = CBAM(256)
        self.sr_net = EnhancedEDSR(scale=scale)
        print(f"初始化SR模型 (单尺度-仅Layer3), 缩放比例: {scale}x")

    def forward(self, lr_img):
        # 记录输入的形状以便调试
        input_shape = lr_img.shape
        
        with torch.no_grad():
            semantic_out = self.semantic_model(lr_img)['out']
        semantic_labels = semantic_out.argmax(dim=1)
        semantic_feature = self.embedding(semantic_labels).permute(0, 3, 1, 2)

        x = self.resnet.conv1(lr_img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        layer1_out = self.resnet.layer1(x)
        layer2_out = self.resnet.layer2(layer1_out)
        layer3_out = self.resnet.layer3(layer2_out)
        
        # 只使用layer3特征，不进行多尺度融合
        semantic_feature_resize = F.interpolate(semantic_feature, size=layer3_out.shape[2:], 
                                              mode='bilinear', align_corners=False)
        
        # 融合语义特征和单层特征
        fused_feature = self.fusion_conv(torch.cat([semantic_feature_resize, layer3_out], dim=1))
        
        sr_out = self.sr_net(self.cbam(fused_feature))

        # 确保输出尺寸是输入的4倍
        if sr_out.shape[2] != input_shape[2] * 4 or sr_out.shape[3] != input_shape[3] * 4:
            sr_out = F.interpolate(sr_out, size=(input_shape[2] * 4, input_shape[3] * 4),
                                   mode='bicubic', align_corners=False)

        return sr_out