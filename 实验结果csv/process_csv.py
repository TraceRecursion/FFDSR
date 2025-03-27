import os
import pandas as pd
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import io

# 定义路径
input_dir = '/Users/sydg/VSCode/FFDSR/实验结果csv/处理前'
output_dir = '/Users/sydg/VSCode/FFDSR/实验结果csv/处理后'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 函数：从文件名中提取实验类型和指标类型
def extract_info(filename):
    # 处理 run-run_ 开头的文件
    if filename.startswith('run-run_'):
        match = re.match(r'run-run_(\d+)-(\d+)-tag-(.+)\.csv', filename)
        if match:
            # 对于 run-run_ 开头的文件，实验类型就是 "run"
            experiment_type = "run"
            metric_type = match.group(3)
            return experiment_type, metric_type
    else:
        # 原有的正则表达式匹配其他文件模式
        match = re.match(r'run-([^_]+_*[^_]+)_run_.*-tag-(.+)\.csv', filename)
        if match:
            experiment_type = match.group(1)
            metric_type = match.group(2)
            return experiment_type, metric_type
    
    # 如果都不匹配，打印调试信息
    print(f"警告: 无法解析文件名 {filename}")
    return None, None

# 添加函数：获取所有未处理的文件
def get_unprocessed_files(all_files, processed_files):
    unprocessed = set(all_files) - set(processed_files)
    if unprocessed:
        print(f"\n未处理的文件 ({len(unprocessed)}):")
        for f in sorted(unprocessed):
            print(f"  - {f}")
    return unprocessed

# 函数：读取CSV文件并跳过注释行
def read_csv_skip_comments(file_path):
    # 读取文件内容
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 过滤掉以'//'开头的行
    filtered_lines = [line for line in lines if not line.strip().startswith('//')]
    
    # 将过滤后的内容转换为字符串
    csv_content = ''.join(filtered_lines)
    
    # 使用pandas从字符串中读取CSV
    return pd.read_csv(io.StringIO(csv_content))

# 扫描处理前目录中的所有CSV文件
all_csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
processed_csv_files = []

# 分组CSV文件
file_groups = defaultdict(list)
for filename in all_csv_files:
    experiment_type, metric_type = extract_info(filename)
    if experiment_type and metric_type:
        key = (experiment_type, metric_type)
        file_groups[key].append(os.path.join(input_dir, filename))
        processed_csv_files.append(filename)

# 列出所有实验类型和指标类型
print("\n识别到的实验类型和指标类型:")
for (exp_type, metric_type), files in file_groups.items():
    print(f"  - {exp_type} - {metric_type}: {len(files)}个文件")

# 检查未处理的文件
get_unprocessed_files(all_csv_files, processed_csv_files)

# 处理每组文件
for (experiment_type, metric_type), file_list in file_groups.items():
    print(f"\n处理 {experiment_type} - {metric_type}, 文件数量: {len(file_list)}")
    
    # 存储所有数据帧
    all_dfs = []
    
    # 读取所有CSV文件
    for file_path in file_list:
        try:
            # 使用自定义函数读取CSV文件并跳过注释行
            df = read_csv_skip_comments(file_path)
            # 确保数据帧包含必要的列
            if 'Step' in df.columns and 'Value' in df.columns:
                all_dfs.append(df)
            else:
                print(f"警告: {file_path} 不包含必要的列")
        except Exception as e:
            print(f"无法读取 {file_path}: {e}")
    
    if not all_dfs:
        print(f"没有可用的数据用于 {experiment_type} - {metric_type}")
        continue
    
    # 找到最小的最大步骤，以确保所有数据都对齐
    max_steps = min(df['Step'].max() for df in all_dfs)
    
    # 准备存储平均值的数据帧
    merged_df = pd.DataFrame()
    merged_df['Step'] = np.arange(0, max_steps + 1)
    
    # 提取每个步骤的值并计算平均值
    values = []
    for step in merged_df['Step']:
        step_values = []
        for df in all_dfs:
            # 找到当前步骤的值
            val_row = df[df['Step'] == step]
            if not val_row.empty:
                step_values.append(val_row['Value'].values[0])
        
        # 计算当前步骤的平均值
        if step_values:
            values.append(np.mean(step_values))
        else:
            values.append(np.nan)
    
    merged_df['Value'] = values
    
    # 保存平均结果到CSV
    output_file = os.path.join(output_dir, f"{experiment_type}_avg_{metric_type}.csv")
    
    # 保存第一个文件的Wall time列（如果有）
    if 'Wall time' in all_dfs[0].columns:
        merged_df['Wall time'] = all_dfs[0]['Wall time'].values[:len(merged_df)]
    
    # 保存到CSV，添加文件路径注释
    with open(output_file, 'w') as f:
        f.write(f"// filepath: {output_file}\n")
        merged_df.to_csv(f, index=False)
    
    print(f"已保存平均结果到 {output_file}")
    
    # 创建可视化
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['Step'], merged_df['Value'])
    
    # 使用英文标题避免中文字体问题
    plt.title(f"{experiment_type} - {metric_type} Average")
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.grid(True)
    
    # 保存图表
    plot_file = os.path.join(output_dir, f"{experiment_type}_avg_{metric_type}.png")
    plt.savefig(plot_file)
    plt.close()

print("\n所有CSV文件处理完成！")
