import torch.nn as nn
from torchvision import models

def get_semantic_model(num_classes=19, pretrained_weights="COCO_WITH_VOC_LABELS_V1"):
    model = models.segmentation.deeplabv3_resnet101(weights=pretrained_weights)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model