import os
import cv2

# 输入和输出路径
hr_img_root = '/Users/sydg/Documents/数据集/DIV2K/train/DIV2K_train_HR'  # 高分辨率图像路径
lr_img_root = '/Users/sydg/Documents/数据集/DIV2K/train/DIV2K_train_LR_bicubic/X4'  # 生成的 X4 低分辨率图像路径

# 创建输出目录
os.makedirs(lr_img_root, exist_ok=True)

# 下采样函数
def downsample_image(image_path, output_path, scale=4):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    lr_height, lr_width = height // scale, width // scale
    lr_img = cv2.resize(img, (lr_width, lr_height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, lr_img)

# 处理所有高分辨率图像
for filename in os.listdir(hr_img_root):
    if filename.endswith('.png'):
        hr_path = os.path.join(hr_img_root, filename)
        lr_path = os.path.join(lr_img_root, filename)
        downsample_image(hr_path, lr_path, scale=4)
        print(f"Generated LR image: {lr_path}")