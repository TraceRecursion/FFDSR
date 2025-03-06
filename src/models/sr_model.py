import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .semantic_model import get_semantic_model

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
    def __init__(self, in_channels=256, out_channels=3, num_blocks=32):
        super(EnhancedEDSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        ) for _ in range(num_blocks)])
        self.conv_up = nn.Conv2d(64, 64, 3, padding=1)
        self.cbam = CBAM(64)
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.body(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.conv_up(x)
        x = self.cbam(x)
        return torch.sigmoid(self.conv_out(x))

class FeatureFusionSR(nn.Module):
    def __init__(self, semantic_model_path=None):
        super(FeatureFusionSR, self).__init__()
        self.semantic_model = get_semantic_model()
        if semantic_model_path:
            self.semantic_model.load_state_dict(torch.load(semantic_model_path, map_location='cpu'))
        self.semantic_model.eval()
        for param in self.semantic_model.parameters():
            param.requires_grad = False

        self.embedding = nn.Embedding(19, 64)
        self.resnet = models.resnet50(weights='DEFAULT')
        self.fusion_conv = nn.Conv2d(1856, 256, 1)
        self.cbam = CBAM(256)
        self.sr_net = EnhancedEDSR()

    def forward(self, lr_img):
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
        return self.sr_net(self.cbam(fused_feature))