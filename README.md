# FFDSR - 特征融合深度超分辨率

基于语义分割特征融合的深度超分辨率重建系统，支持城市场景图像的语义分析和高质量重建。

## 项目结构
```
FFDSR/
├── configs/                  # 配置文件目录
│   ├── semantic_train.yaml   # 语义分割训练配置
│   ├── semantic_test.yaml    # 语义分割测试配置
│   ├── semantic_eval.yaml    # 语义分割评估配置
│   ├── sr_train.yaml         # 超分辨率训练配置
│   └── sr_test.yaml          # 超分辨率测试配置
├── date_X4/                  # 数据预处理脚本
│   ├── CitySpaces_HR-to-LR.py  # CitySpaces数据集下采样处理
│   └── DIV2K_HR-to-LR.py     # DIV2K数据集下采样处理
├── models/                   # 预训练模型保存目录
│   └── best_deeplabv3plus_mobilenet_cityscapes_os16.pth  # DeepLabV3+预训练权重
├── src/                      # 源代码目录
│   ├── datasets/             # 数据集模块
│   │   ├── __init__.py
│   │   ├── semantic_dataset.py  # 语义分割数据集
│   │   └── sr_dataset.py     # 超分辨率数据集
│   ├── models/               # 模型模块
│   │   ├── __init__.py
│   │   ├── _deeplab.py       # DeepLab框架实现
│   │   ├── backbone/         # 骨干网络
│   │   │   ├── __init__.py
│   │   │   └── mobilenetv2.py  # MobileNetV2实现
│   │   ├── network.py        # 网络构建函数
│   │   ├── semantic_model.py # 语义分割模型
│   │   ├── sr_model.py       # 超分辨率模型
│   │   └── utils.py          # 模型工具函数
│   ├── utils/                # 工具模块
│   │   ├── __init__.py
│   │   ├── common.py         # 通用工具函数
│   │   ├── config.py         # 配置文件解析
│   │   ├── loss.py           # 损失函数
│   │   ├── metrics.py        # 评估指标
│   │   └── visualize.py      # 可视化函数
│   ├── trainers/             # 训练和测试逻辑
│   │   ├── __init__.py
│   │   ├── semantic_trainer.py  # 语义分割训练器
│   │   ├── semantic_tester.py   # 语义分割测试器
│   │   ├── sr_trainer.py     # 超分辨率训练器
│   │   └── sr_tester.py      # 超分辨率测试器
│   ├── __init__.py
│   └── main.py               # 主入口脚本
├── outputs/                  # 输出目录（自动生成）
│   └── semantic/             # 语义分割结果
├── runs/                     # 超分辨率训练日志目录
└── runs_semantic/            # 语义分割训练日志目录
```

## 使用方法

安装依赖：

```bash
pip install torch torchvision torchmetrics pyyaml tqdm pillow numpy
```

运行任务：

语义分割训练：

```bash
python -m src.main --config configs/semantic_train.yaml
```

语义分割测试：

```bash
python -m src.main --config configs/semantic_test.yaml
```

语义分割评估（IoU分析）：

```bash
python -m src.main --config configs/semantic_eval.yaml
```

超分辨率训练：

```bash
python -m src.main --config configs/sr_train.yaml
```

超分辨率测试：

```bash
python -m src.main --config configs/sr_test.yaml
```

调整参数：

修改 configs/ 中的 YAML 文件以调整超参数。

tensorboard：

```bash
tensorboard --logdir=runs --host=100.74.5.7 --port=6006
tensorboard --logdir=runs_semantic --host=100.74.5.7 --port=6006
```