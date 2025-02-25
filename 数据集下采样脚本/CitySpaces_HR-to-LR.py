import os
import cv2

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

base_data_dir = os.path.join(current_dir, '../../../Documents/数据集')
cityscapes_base = os.path.join(base_data_dir, 'CitySpaces/leftImg8bit_trainvaltest/leftImg8bit')
hr_img_root = os.path.join(cityscapes_base, 'train')
lr_img_root = os.path.join(cityscapes_base, 'train_lr_x4')

# 添加路径验证
if not os.path.exists(hr_img_root):
    raise FileNotFoundError(f"HR图像路径不存在: {hr_img_root}")
if not os.path.exists(lr_img_root):
    raise FileNotFoundError(f"LR图像路径不存在: {lr_img_root}")

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