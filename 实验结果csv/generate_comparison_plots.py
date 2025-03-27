import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# 定义路径
input_dir = '/Users/sydg/VSCode/FFDSR/实验结果csv/处理后'
output_dir = '/Users/sydg/VSCode/FFDSR/实验结果csv/对比图'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 要比较的模型列表
models = ['no_cbam', 'no_semantic', 'run', 'single_scale', 'srcnn']

# 要生成的指标列表
metrics = ['Learning_Rate', 'Loss_Train', 'Loss_Val', 'PSNR_Val', 'SSIM_Val']

# 为每个模型定义颜色和线型
colors = {
    'no_cbam': 'red',
    'no_semantic': 'blue',
    'run': 'green',
    'single_scale': 'purple',
    'srcnn': 'orange'
}

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

# 为每个指标创建一个图表
for metric in metrics:
    plt.figure(figsize=(12, 8))
    
    # 为每个模型绘制一条线
    for model in models:
        csv_file = os.path.join(input_dir, f"{model}_avg_{metric}.csv")
        
        if not os.path.exists(csv_file):
            print(f"警告: 找不到文件 {csv_file}")
            continue
        
        # 使用自定义函数读取CSV文件
        try:
            df = read_csv_skip_comments(csv_file)
            
            # 绘制曲线
            plt.plot(df['Step'], df['Value'], color=colors[model], linewidth=2, label=model)
        except Exception as e:
            print(f"无法处理文件 {csv_file}: {e}")
    
    # 添加图表标题和标签
    plt.title(f"{metric} Comparison", fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(fontsize=12)
    
    # 根据指标调整Y轴
    if metric == 'Learning_Rate':
        plt.yscale('log')  # 使用对数刻度更好地显示学习率变化
        plt.ylabel('Learning Rate (log scale)', fontsize=14)
    elif metric == 'PSNR_Val':
        plt.ylabel('PSNR (dB)', fontsize=14)
    
    # 保存图表
    output_file = os.path.join(output_dir, f"comparison_{metric}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存 {metric} 对比图到 {output_file}")

# 创建一个HTML文件，便于查看所有图表
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .image-container { margin-bottom: 30px; }
        img { max-width: 100%; border: 1px solid #ddd; }
        h2 { color: #444; }
    </style>
</head>
<body>
    <h1>Model Comparison</h1>
"""

for metric in metrics:
    img_path = f"comparison_{metric}.png"
    html_content += f"""
    <div class="image-container">
        <h2>{metric.replace('_', ' ')} Comparison</h2>
        <img src="{img_path}" alt="{metric} comparison">
    </div>
    """

html_content += """
</body>
</html>
"""

# 保存HTML文件
html_file = os.path.join(output_dir, 'comparison_results.html')
with open(html_file, 'w') as f:
    f.write(html_content)

print(f"\n已生成HTML报告: {html_file}")
print(f"\n所有对比图已生成完成，保存在 {output_dir} 目录中")
