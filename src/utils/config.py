import yaml
import os


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def resolve_paths(config, project_root):
    # 获取 base_dir
    base_dir = config['data'].get('base_dir', '')
    print(f"原始 project_root: {project_root}")
    print(f"原始 base_dir: {base_dir}")

    # 手动解析相对路径
    current_path = project_root
    for part in base_dir.split(os.sep):
        if part == '..':
            current_path = os.path.dirname(current_path)
        elif part and part != '.':
            current_path = os.path.join(current_path, part)
    resolved_base_dir = os.path.normpath(current_path)
    print(f"解析后的 base_dir: {resolved_base_dir}")

    # 对非 base_dir 的 dir 字段拼接 resolved_base_dir
    for key, value in config['data'].items():
        if isinstance(value, str) and 'dir' in key and key != 'base_dir':
            config['data'][key] = os.path.normpath(os.path.join(resolved_base_dir, value))

    # 处理输出目录
    if 'training' in config and 'output_dir' in config['training']:
        config['training']['output_dir'] = os.path.normpath(
            os.path.join(project_root, config['training']['output_dir']))
    if 'training' in config and 'log_dir' in config['training']:
        config['training']['log_dir'] = os.path.normpath(os.path.join(project_root, config['training']['log_dir']))
    if 'output' in config and 'dir' in config['output']:
        config['output']['dir'] = os.path.normpath(os.path.join(project_root, config['output']['dir']))

    return config