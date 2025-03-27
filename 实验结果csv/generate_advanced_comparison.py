import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import io

# 设置Matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

# 定义路径
input_dir = '/Users/sydg/VSCode/FFDSR/实验结果csv/处理后'
output_dir = '/Users/sydg/VSCode/FFDSR/实验结果csv/高级对比图'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

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

# 要比较的模型列表
models = ['no_cbam', 'no_semantic', 'run', 'single_scale', 'srcnn']

# 要生成的指标列表
metrics = ['Learning_Rate', 'Loss_Train', 'Loss_Val', 'PSNR_Val', 'SSIM_Val']

# 模型的显示名称（可选，更美观的名称）
model_display_names = {
    'no_cbam': 'No CBAM',
    'no_semantic': 'No Semantic',
    'run': 'Base Model',
    'single_scale': 'Single Scale',
    'srcnn': 'SRCNN'
}

# 为每个模型定义颜色和线型
colors = {
    'no_cbam': '#FF5733',       # 红色
    'no_semantic': '#3498DB',   # 蓝色
    'run': '#2ECC71',           # 绿色
    'single_scale': '#9B59B6',  # 紫色
    'srcnn': '#F39C12'          # 橙色
}

linestyles = {
    'no_cbam': '-',
    'no_semantic': '-',
    'run': '-',
    'single_scale': '-',
    'srcnn': '-'
}

markers = {
    'no_cbam': 'o',
    'no_semantic': 's',
    'run': '^',
    'single_scale': 'D',
    'srcnn': 'x'
}

# 指标的美化名称
metric_display_names = {
    'Learning_Rate': 'Learning Rate',
    'Loss_Train': 'Training Loss',
    'Loss_Val': 'Validation Loss',
    'PSNR_Val': 'PSNR (dB)',
    'SSIM_Val': 'SSIM'
}

# 为每个指标创建一个图表
for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    max_steps = 0
    all_values = []  # 用于收集所有模型的值
    
    # 为每个模型绘制一条线
    for model in models:
        csv_file = os.path.join(input_dir, f"{model}_avg_{metric}.csv")
        
        if not os.path.exists(csv_file):
            print(f"警告: 找不到文件 {csv_file}")
            continue
        
        try:
            # 读取CSV文件
            df = read_csv_skip_comments(csv_file)
            
            # 收集所有值以便后续计算适当的Y轴范围
            all_values.extend(df['Value'].tolist())
            
            # 获取最大步数
            if df['Step'].max() > max_steps:
                max_steps = df['Step'].max()
            
            # 每10个点采样一次，避免图表过于密集
            sampling_rate = max(1, len(df) // 100)
            sampled_df = df.iloc[::sampling_rate]
            
            # 绘制曲线
            plt.plot(df['Step'], df['Value'], 
                    color=colors[model], 
                    linestyle=linestyles[model], 
                    linewidth=2, 
                    label=model_display_names.get(model, model),
                    alpha=0.9)
            
            # 在曲线上添加少量标记点
            marker_sampling_rate = max(1, len(df) // 10)
            plt.plot(df['Step'].iloc[::marker_sampling_rate], 
                    df['Value'].iloc[::marker_sampling_rate], 
                    color=colors[model], 
                    marker=markers[model], 
                    linestyle='none', 
                    markersize=6,
                    alpha=0.7)
        except Exception as e:
            print(f"无法处理文件 {csv_file}: {e}")
    
    # 添加图表标题和标签
    plt.title(f"{metric_display_names.get(metric, metric)} Comparison", fontsize=18, pad=20)
    plt.xlabel('Training Steps', fontsize=16)
    plt.ylabel(metric_display_names.get(metric, metric), fontsize=16)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 根据指标调整Y轴
    if metric == 'Learning_Rate':
        plt.yscale('log')  # 使用对数刻度更好地显示学习率变化
    elif metric == 'Loss_Train' or metric == 'Loss_Val':
        # 根据所有模型的数据计算合适的Y轴范围
        if all_values:
            # 移除异常值（可选）
            q1, q3 = np.percentile(all_values, [25, 75])
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            
            # 设置Y轴上限为上界或所有值的95百分位数，取较小者
            max_y = min(upper_bound, np.percentile(all_values, 95))
            
            # 如果上限太小，使用当前自动计算的上限
            current_ymin, current_ymax = ax.get_ylim()
            if max_y < current_ymax / 10:
                max_y = current_ymax
                
            # 设置新的Y轴范围
            ax.set_ylim(0, max_y)
            
            # 添加注释说明范围已经调整
            plt.annotate(f'Y-axis limited to improve visibility', 
                        xy=(0.02, 0.98), 
                        xycoords='axes fraction', 
                        fontsize=10, 
                        alpha=0.7,
                        va='top')
    
    # X轴使用整数刻度
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加图例
    legend = plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9, shadow=True)
    
    # 保存图表
    output_file = os.path.join(output_dir, f"comparison_{metric}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # 再保存一个PDF版本以便高质量打印
    pdf_file = os.path.join(output_dir, f"comparison_{metric}.pdf")
    plt.savefig(pdf_file, bbox_inches='tight')
    
    # 如果是Loss类型的图表，另外保存一个不限制Y轴的版本
    if metric == 'Loss_Train' or metric == 'Loss_Val':
        # 重置Y轴
        ax.relim()
        ax.autoscale_view()
        
        # 更新注释
        for txt in ax.texts:
            txt.set_visible(False)
        
        plt.annotate(f'Full Y-axis range', 
                    xy=(0.02, 0.98), 
                    xycoords='axes fraction', 
                    fontsize=10, 
                    alpha=0.7,
                    va='top')
        
        # 保存完整范围版本
        full_range_file = os.path.join(output_dir, f"comparison_{metric}_full_range.png")
        plt.savefig(full_range_file, dpi=300, bbox_inches='tight')
        print(f"已保存完整范围版本到 {full_range_file}")
    
    plt.close(fig)
    
    print(f"已保存 {metric} 对比图到 {output_file}")

# 创建一个HTML文件，便于查看所有图表
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Model Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; background-color: #f5f5f5; }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .image-container { margin-bottom: 40px; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        img { max-width: 100%; display: block; margin: 0 auto; }
        h2 { color: #34495e; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top: 0; }
        .description { color: #7f8c8d; margin-bottom: 15px; }
        footer { text-align: center; margin-top: 40px; color: #95a5a6; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Advanced Model Comparison</h1>
"""

descriptions = {
    'Learning_Rate': 'Comparison of learning rate schedules used during training. Note the logarithmic scale.',
    'Loss_Train': 'Training loss comparison across different models. Lower values indicate better training convergence.',
    'Loss_Val': 'Validation loss comparison, showing how well each model generalizes to unseen data.',
    'PSNR_Val': 'Peak Signal-to-Noise Ratio (PSNR) on validation data. Higher values indicate better image quality.',
    'SSIM_Val': 'Structural Similarity Index (SSIM) on validation data. Higher values indicate better perceptual quality.'
}

for metric in metrics:
    img_path = f"comparison_{metric}.png"
    description = descriptions.get(metric, "")
    
    html_content += f"""
    <div class="image-container">
        <h2>{metric_display_names.get(metric, metric)} Comparison</h2>
        <p class="description">{description}</p>
        <img src="{img_path}" alt="{metric} comparison">
    </div>
    """

html_content += """
    <footer>
        <p>Generated using matplotlib. All models trained with the same dataset and evaluation metrics.</p>
    </footer>
</body>
</html>
"""

# 保存HTML文件
html_file = os.path.join(output_dir, 'advanced_comparison_results.html')
with open(html_file, 'w') as f:
    f.write(html_content)

print(f"\n已生成HTML报告: {html_file}")
print(f"\n所有高级对比图已生成完成，保存在 {output_dir} 目录中")
