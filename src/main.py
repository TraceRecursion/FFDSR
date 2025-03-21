import argparse
import os
from src.utils.config import load_config, resolve_paths
from src.trainers.semantic_trainer import SemanticTrainer
from src.trainers.semantic_tester import SemanticTester
from src.trainers.semantic_evaluator import SemanticEvaluator
from src.trainers.sr_trainer import SRTrainer
from src.trainers.sr_tester import SRTester
from src.trainers.srcnn_trainer import SRCNNTrainer
from src.trainers.srcnn_tester import SRCNNTester
import sys


def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation and Super-Resolution Training/Testing")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # 输出运行环境信息，有助于调试
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"配置文件: {args.config}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(project_root, args.config))
    config = resolve_paths(config, project_root)

    print(f"加载配置: {config['task']}")

    if 'training' in config and 'output_dir' in config['training']:
        os.makedirs(config['training']['output_dir'], exist_ok=True)
    if 'output' in config and 'dir' in config['output']:
        os.makedirs(config['output']['dir'], exist_ok=True)

    # 添加消融实验的任务类型
    ablation_train_tasks = ['sr_train_no_semantic', 'sr_train_no_cbam', 'sr_train_single_scale']
    ablation_test_tasks = ['sr_test_no_semantic', 'sr_test_no_cbam', 'sr_test_single_scale']
    
    try:
        if config['task'] == 'semantic_train':
            print("启动语义分割训练...")
            trainer = SemanticTrainer(config)
            trainer.train()
        elif config['task'] == 'semantic_test':
            print("启动语义分割测试...")
            tester = SemanticTester(config)
            tester.test()
        elif config['task'] == 'semantic_eval':
            print("启动语义分割评估...")
            evaluator = SemanticEvaluator(config)
            evaluator.evaluate()
        elif config['task'] == 'sr_train' or config['task'] in ablation_train_tasks:
            print(f"启动超分辨率训练 (模式: {config['task']})...")
            trainer = SRTrainer(config)
            trainer.train()
        elif config['task'] == 'sr_test' or config['task'] in ablation_test_tasks:
            print(f"启动超分辨率测试 (模式: {config['task']})...")
            tester = SRTester(config)
            tester.test()
        elif config['task'] == 'srcnn_train':
            print("启动SRCNN训练...")
            trainer = SRCNNTrainer(config)
            trainer.train()
        elif config['task'] == 'srcnn_test':
            print("启动SRCNN测试...")
            tester = SRCNNTester(config)
            tester.test()
        else:
            print(f"错误: 未知任务类型 '{config['task']}'")
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()