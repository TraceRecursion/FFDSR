import torch
from torch.utils.data import DataLoader
from ..datasets.semantic_dataset import CityscapesDataset
from ..models.semantic_model import get_semantic_model
from ..utils.visualize import visualize_segmentation
import os
from PIL import Image  # 修复错误1：显式导入 Image

class SemanticTester:
    def __init__(self, config):
        self.config = config
        self.device, device_name = torch.cuda.is_available() and (torch.device("cuda"), "CUDA") or (torch.device("cpu"), "CPU")
        print(f"使用设备: {device_name}")
        self.model = get_semantic_model(self.config['model']['num_classes'], self.config['model']['pretrained_weights']).to(self.device)
        self.model.load_state_dict(torch.load(self.config['model']['model_path'], map_location=self.device))
        self.model.eval()

    def test(self):
        # 修改数据集调用以返回原始图像
        dataset = CityscapesDataset(
            self.config['data']['test_img_dir'],
            self.config['data']['test_label_dir'],
            self.config['data']['crop_size']
        )
        # 自定义 collate_fn 以返回原始图像
        def collate_fn(batch):
            img_tensors, label_nps, img_names = zip(*batch)
            orig_imgs = [Image.open(dataset.images[i]).convert('RGB') for i in range(len(img_names))]
            return torch.stack(img_tensors), label_nps, orig_imgs, img_names

        loader = DataLoader(
            dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
        os.makedirs(self.config['output']['dir'], exist_ok=True)

        with torch.no_grad():
            for img_tensor, label_np, orig_imgs, img_names in loader:
                img_tensor = img_tensor.to(self.device)
                outputs = self.model(img_tensor)['out']
                pred = outputs.argmax(dim=1).cpu().numpy()[0]  # 假设 batch_size=1
                result_img = visualize_segmentation(pred, orig_imgs[0])  # 使用返回的原始图像
                result_img.save(os.path.join(self.config['output']['dir'], f"pred_{img_names[0]}"))
                print(f"保存结果: {os.path.join(self.config['output']['dir'], f'pred_{img_names[0]}')}")