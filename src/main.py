import argparse
import os
from src.utils.config import load_config, resolve_paths
from src.trainers.semantic_trainer import SemanticTrainer
from src.trainers.semantic_tester import SemanticTester
from src.trainers.semantic_evaluator import SemanticEvaluator
from src.trainers.sr_trainer import SRTrainer
from src.trainers.sr_tester import SRTester
from src.trainers.srcnn_trainer import SRCNNTrainer  # 导入SRCNN训练器
from src.trainers.srcnn_tester import SRCNNTester    # 导入SRCNN测试器


def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation and Super-Resolution Training/Testing")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(os.path.join(project_root, args.config))
    config = resolve_paths(config, project_root)

    if 'training' in config and 'output_dir' in config['training']:
        os.makedirs(config['training']['output_dir'], exist_ok=True)
    if 'output' in config and 'dir' in config['output']:
        os.makedirs(config['output']['dir'], exist_ok=True)

    if config['task'] == 'semantic_train':
        trainer = SemanticTrainer(config)
        trainer.train()
    elif config['task'] == 'semantic_test':
        tester = SemanticTester(config)
        tester.test()
    elif config['task'] == 'semantic_eval':
        evaluator = SemanticEvaluator(config)
        evaluator.evaluate()
    elif config['task'] == 'sr_train':
        trainer = SRTrainer(config)
        trainer.train()
    elif config['task'] == 'sr_test':
        tester = SRTester(config)
        tester.test()
    elif config['task'] == 'srcnn_train':  # 新增SRCNN训练任务
        trainer = SRCNNTrainer(config)
        trainer.train()
    elif config['task'] == 'srcnn_test':   # 新增SRCNN测试任务
        tester = SRCNNTester(config)
        tester.test()


if __name__ == "__main__":
    main()