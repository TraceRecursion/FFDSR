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
                # 首先尝试使用weights_only=False加载
                print("尝试使用weights_only=False加载...")
                try:
                    # 在PyTorch 2.0+中，为了安全，torch.load默认使用weights_only=True
                    checkpoint = torch.load(pretrained_weights, map_location=device, weights_only=False)

                    # 处理检查点字典的情况
                    if isinstance(checkpoint, dict):
                        # 检查是否有model_state键
                        if "model_state" in checkpoint:
                            print("检测到模型状态存储在'model_state'键中")
                            checkpoint = checkpoint["model_state"]
                        # 检查常见的其他键名
                        elif "state_dict" in checkpoint:
                            print("检测到模型状态存储在'state_dict'键中")
                            checkpoint = checkpoint["state_dict"]

                    # 尝试直接加载权重
                    try:
                        model.load_state_dict(checkpoint)
                        print("预训练权重加载成功")
                    except Exception as e:
                        print(f"直接加载失败: {e}")
                        print("尝试部分加载权重...")
                        # 尝试加载匹配的键
                        model_dict = model.state_dict()

                        # 过滤掉不匹配的键
                        pretrained_dict = {k: v for k, v in checkpoint.items()
                                           if k in model_dict and v.shape == model_dict[k].shape}

                        print(f"找到 {len(pretrained_dict)}/{len(model_dict)} 个匹配的参数")
                        if len(pretrained_dict) > 0:
                            # 更新模型状态字典
                            model_dict.update(pretrained_dict)
                            model.load_state_dict(model_dict)
                            print(f"部分加载成功，加载了 {len(pretrained_dict)}/{len(model_dict)} 层")
                        else:
                            print("没有找到匹配的参数，尝试其他方法")
                            raise ValueError("没有匹配的参数")

                except Exception as e:
                    print(f"使用weights_only=False加载失败: {e}")

                    # 尝试使用安全方式加载
                    print("尝试使用安全方式加载...")
                    try:
                        checkpoint = torch.load(pretrained_weights, map_location=device, weights_only=True)
                        model.load_state_dict(checkpoint)
                        print("预训练权重加载成功(安全模式)")
                    except Exception as e2:
                        print(f"直接加载失败: {e2}")
                        # 最后尝试旧版本PyTorch兼容加载
                        try:
                            checkpoint = torch.load(pretrained_weights, map_location=device)
                            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                                checkpoint = checkpoint['model_state']
                            model_dict = model.state_dict()
                            pretrained_dict = {k: v for k, v in checkpoint.items()
                                               if k in model_dict and v.shape == model_dict[k].shape}
                            if len(pretrained_dict) > 0:
                                model_dict.update(pretrained_dict)
                                model.load_state_dict(model_dict)
                                print(f"兼容模式部分加载成功，加载了 {len(pretrained_dict)}/{len(model_dict)} 层")
                            else:
                                print("所有加载方法都失败了，将使用随机初始化的模型")
                        except Exception as e3:
                            print(f"所有加载方法都失败了: {e3}")
                            print("将使用随机初始化的模型")
            except Exception as e:
                print(f"加载预训练权重时出现意外错误: {e}")
                print("将使用随机初始化的模型")
        else:
            print(f"警告: 预训练权重文件不存在 {pretrained_weights}")

    return model