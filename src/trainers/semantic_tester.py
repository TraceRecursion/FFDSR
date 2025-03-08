import torch
from torch.utils.data import DataLoader
from ..datasets.semantic_dataset import CityscapesDataset
from ..models.semantic_model import get_semantic_model
from ..utils.visualize import visualize_segmentation
import os
from PIL import Image
from tqdm import tqdm


class SemanticTester:
    def __init__(self, config):
        self.config = config
        self.device, device_name = torch.cuda.is_available() and (torch.device("cuda"), "CUDA") or (
        torch.device("cpu"), "CPU")
        print(f"使用设备: {device_name}")
        self.model = get_semantic_model(self.config['model']['num_classes'],
                                        self.config['model']['pretrained_weights']).to(self.device)

        checkpoint_path = self.config['model']['model_path']
        print(f"加载模型权重: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"警告: 模型权重文件不存在: {checkpoint_path}")

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        print("模型权重加载成功")
        self.model.eval()

    def test(self):
        dataset = CityscapesDataset(
            self.config['data']['test_img_dir'],
            self.config['data']['test_label_dir'],
            self.config['data']['crop_size'],
            test_mode=True
        )

        print(f"测试数据集大小: {len(dataset)}张图像")
        os.makedirs(self.config['output']['dir'], exist_ok=True)

        progress_bar = tqdm(range(len(dataset)), desc="语义分割测试")

        for idx in progress_bar:
            img_tensor, label_tensor, img_name = dataset[idx]
            img_path = dataset.images[idx]

            progress_bar.set_description(f"处理: {img_name}")

            orig_img = Image.open(img_path).convert('RGB')
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)['out']
                pred = outputs.argmax(dim=1).cpu().numpy()[0]

                result_img = visualize_segmentation(pred, orig_img)
                save_path = os.path.join(self.config['output']['dir'], f"pred_{img_name}")
                result_img.save(save_path)

        print(f"语义分割测试完成! 结果保存至: {self.config['output']['dir']}")