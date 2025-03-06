tensorboard --logdir=runs_semantic --host=100.74.5.7 --port=6006

FFDSR/
├── configs/                  # 配置文件目录
│   ├── semantic_train.yaml   # 语义分割训练配置
│   ├── semantic_test.yaml    # 语义分割测试配置
│   ├── sr_train.yaml         # 超分辨率训练配置
│   └── sr_test.yaml          # 超分辨率测试配置
├── src/                      # 源代码目录
│   ├── datasets/             # 数据集模块
│   │   ├── __init__.py
│   │   ├── semantic_dataset.py
│   │   └── sr_dataset.py
│   ├── models/               # 模型模块
│   │   ├── __init__.py
│   │   ├── semantic_model.py
│   │   └── sr_model.py
│   ├── utils/                # 工具模块
│   │   ├── __init__.py
│   │   ├── config.py       # 配置文件解析
│   │   ├── visualize.py    # 可视化函数
│   │   ├── metrics.py      # 评估指标
│   │   └── common.py       # 通用工具函数
│   ├── trainers/             # 训练和测试逻辑
│   │   ├── __init__.py
│   │   ├── semantic_trainer.py
│   │   ├── semantic_tester.py
│   │   ├── sr_trainer.py
│   │   └── sr_tester.py
│   └── main.py               # 主入口脚本
├── outputs/                  # 输出目录（自动生成）
├── runs/                     # TensorBoard日志目录（自动生成）
└── models/                   # 模型保存目录（自动生成）

使用方法

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
