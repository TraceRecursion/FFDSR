import torch.nn as nn
import torch
from .network import deeplabv3plus_mobilenet
import os
from ..utils.common import get_device

def get_semantic_model(num_classes=19, pretrained_weights=None):
    """
    获取语义分割模型
    
    Args:
        num_classes: 分类类别数
        pretrained_weights: 预训练权重路径或标识符
        
    Returns:
        model: DeepLabV3+ 模型
    """
    device, _ = get_device()
    
    # 创建DeepLabV3+模型，使用MobileNetV2作为骨干网络
    try:
        # 首先尝试使用自定义的output_stride参数
        model = deeplabv3plus_mobilenet(num_classes=num_classes, output_stride=16, pretrained_backbone=True)
    except TypeError as e:
        print(f"警告: 无法使用output_stride参数: {e}")
        # 如果失败，尝试不使用output_stride参数
        model = deeplabv3plus_mobilenet(num_classes=num_classes, pretrained_backbone=True)
    
    # 如果提供了预训练权重路径
    if pretrained_weights is not None and pretrained_weights != "COCO_WITH_VOC_LABELS_V1":
        if os.path.exists(pretrained_weights):
            print(f"加载预训练权重: {pretrained_weights}")
            try:
                # 在PyTorch 2.6+中，为了安全，torch.load默认使用weights_only=True
                # 这会导致一些预训练权重无法加载，我们需要显式地设置weights_only=False
                checkpoint = torch.load(pretrained_weights, map_location=device, weights_only=False)
                
                # 有些预训练权重可能会有前缀，我们需要处理一下
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    checkpoint = checkpoint["model_state"]
                    
                # 尝试加载权重
                try:
                    model.load_state_dict(checkpoint)
                    print("预训练权重加载成功")
                except Exception as e:
                    print(f"加载预训练权重出错: {e}")
                    print("尝试部分加载权重...")
                    # 尝试加载匹配的键
                    model_dict = model.state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    print(f"部分加载成功，加载了 {len(pretrained_dict)}/{len(model_dict)} 层")
            except Exception as e:
                print(f"加载预训练权重文件时出错: {e}")
                # 尝试使用安全模式加载
                try:
                    print("尝试使用安全模式加载权重文件...")
                    checkpoint = torch.load(pretrained_weights, map_location=device, weights_only=True)
                    model.load_state_dict(checkpoint)
                    print("预训练权重加载成功(安全模式)")
                except Exception as e2:
                    print(f"安全模式加载失败: {e2}")
        else:
            print(f"警告: 预训练权重文件不存在 {pretrained_weights}")
            
    return model