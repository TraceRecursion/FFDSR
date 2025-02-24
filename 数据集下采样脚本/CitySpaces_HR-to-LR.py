import os
import cv2

# 输入和输出路径
hr_img_root = '/Users/sydg/Documents/数据集/CitySpaces/leftImg8bit_trainvaltest/leftImg8bit/train'
lr_img_root = '/Users/sydg/Documents/数据集/CitySpaces/leftImg8bit_trainvaltest/leftImg8bit/train_lr_x4'

# 下采样函数
def downsample_image(image_path, output_path, scale=4):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    lr_height, lr_width = height // scale, width // scale
    lr_img = cv2.resize(img, (lr_width, lr_height), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, lr_img)

# 遍历所有城市子目录
for city in os.listdir(hr_img_root):
    city_dir = os.path.join(hr_img_root, city)
    if os.path.isdir(city_dir):
        # 创建对应的 lr_x4 子目录
        lr_city_dir = os.path.join(lr_img_root, city)
        os.makedirs(lr_city_dir, exist_ok=True)
        # 处理该城市目录下的所有图像
        for filename in os.listdir(city_dir):
            if filename.endswith('.png'):
                hr_path = os.path.join(city_dir, filename)
                lr_filename = filename.replace('.png', '_lr_x4.png')
                lr_path = os.path.join(lr_city_dir, lr_filename)
                downsample_image(hr_path, lr_path, scale=4)
                print(f"Generated LR image: {lr_path}")