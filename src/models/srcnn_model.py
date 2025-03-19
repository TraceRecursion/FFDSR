import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    """
    经典SRCNN模型实现
    Dong, C., et al. (2014). "Learning a Deep Convolutional Network for Image Super-Resolution."
    """
    def __init__(self, num_channels=3, scale=4):
        super(SRCNN, self).__init__()
        self.scale = scale
        
        # 定义SRCNN的三层架构
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)  # 特征提取层
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)            # 非线性映射层
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)  # 重建层
        
        # 权重初始化
        self._initialize_weights()
        
        print(f"初始化SRCNN模型, 缩放比例: {scale}x")
    
    def forward(self, x):
        # 双三次插值上采样
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
            
        # SRCNN处理
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
