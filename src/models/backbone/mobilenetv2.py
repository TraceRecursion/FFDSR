from torch import nn
from torchvision.models import mobilenet_v2 as tv_mobilenet_v2
from torchvision.models import MobileNet_V2_Weights

__all__ = ['mobilenet_v2']

def mobilenet_v2(pretrained=False, progress=True, output_stride=8, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        output_stride (int): Output stride of the network
    """
    # 使用torchvision的mobilenet_v2函数，使用新的weights参数替代pretrained
    weights = None
    if pretrained:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
    
    # 注意这里使用了重命名后的tv_mobilenet_v2而不是mobilenet_v2
    model = tv_mobilenet_v2(weights=weights, **kwargs)
    
    # 根据output_stride修改模型的步长
    if output_stride == 16:
        model.features[14].conv[1][0].stride = (1, 1)
        model.features[15].conv[1][0].stride = (1, 1)
        model.features[16].conv[1][0].stride = (1, 1)
        model.features[17].conv[1][0].stride = (1, 1)
    elif output_stride == 8:
        model.features[5].conv[1][0].stride = (1, 1)
        model.features[6].conv[1][0].stride = (1, 1)
        model.features[7].conv[1][0].stride = (1, 1)
        model.features[8].conv[1][0].stride = (1, 1)
        model.features[9].conv[1][0].stride = (1, 1)
        model.features[10].conv[1][0].stride = (1, 1)
        model.features[11].conv[1][0].stride = (1, 1)
        model.features[12].conv[1][0].stride = (1, 1)
        model.features[13].conv[1][0].stride = (1, 1)
        model.features[14].conv[1][0].stride = (1, 1)
        model.features[15].conv[1][0].stride = (1, 1)
        model.features[16].conv[1][0].stride = (1, 1)
        model.features[17].conv[1][0].stride = (1, 1)
    
    return model
