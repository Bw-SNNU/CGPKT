import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置字体和图表样式
plt.rcParams.update({
    'font.sans-serif': ['SimHei'],  # 中文字体
    'axes.unicode_minus': False,     # 负号显示
    'font.size': 8,                 # 基础字体大小
    'axes.labelsize': 10,           # 轴标签字体大小改小
    'axes.titlesize': 12,           # 标题字体大小改小
    'xtick.labelsize': 8,           # x轴刻度字体大小改小
    'ytick.labelsize': 8,           # y轴刻度字体大小改小
    'legend.fontsize': 8,           # 图例字体大小改小
    'figure.dpi': 300,              # 图像DPI
    # 'axes.grid': True,              # 显示网格
    'grid.alpha': 0.3,              # 网格透明度
    'grid.linestyle': '--'          # 网格线样式
})

def load_and_process_data(file_numbers):
    all_student_correct_rates = []
    
    for i in file_numbers:
        try:
            df = pd.read_csv(f'student_log_{i}.csv')
            student_correct_rates = df.groupby('student_id')['correct'].mean()
            all_student_correct_rates.extend(student_correct_rates.values)
        except Exception as e:
            print(f"处理文件 student_log_{i}.csv 时出错: {e}")
            continue
    
    return np.array(all_student_correct_rates)

def plot_normal_distribution(correct_rates):
    # 计算统计量
    mean = np.mean(correct_rates)
    std = np.std(correct_rates)
    
    # 创建更小的图形
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # 设置背景色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # 设置x轴范围和刻度
    x_min, x_max = 0, 1
    x = np.linspace(x_min, x_max, 100)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    
    # 绘制直方图
    n, bins, patches = ax.hist(correct_rates, bins=30, density=True, 
                             alpha=0.6, color='#3498db', 
                             edgecolor='black', linewidth=1)
    
    # 绘制正态分布曲线
    y = stats.norm.pdf(x, mean, std)
    ax.plot(x, y, color='#e74c3c', linewidth=2.5)
    
    # 设置字体参数
    font_title = {'family': 'Times New Roman', 'size': 10}
    font_label = {'family': 'Times New Roman', 'size': 10}
    font_tick = {'family': 'Times New Roman', 'size': 8}
    font_text = {'family': 'Times New Roman', 'size': 8}
    
    # 添加标签和标题
    ax.set_xlabel('Correct answer rate', **font_label)
    ax.set_ylabel('Probability Density', **font_label)
    ax.set_title('Distribution of students answer accuracy', **font_title, pad=15)
    
    # 设置刻度字体
    ax.set_xticklabels([f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.1)], **font_tick)
    ax.set_yticklabels([f"{tick:.1f}" for tick in ax.get_yticks()], **font_tick)
    
    # 添加统计信息文本框
    stats_text = f'Number of samples: {len(correct_rates)}\n'
    stats_text += f'Average: {mean:.3f}\n'
    stats_text += f'Standard deviation: {std:.3f}'
    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', 
                     edgecolor='#666666',
                     alpha=0.8,
                     boxstyle='round,pad=0.5'),
            verticalalignment='top',
            horizontalalignment='right',
            fontdict=font_text)
    
    # 去除图例
    # 不添加legend即可
    
    # 设置轴线颜色
    for spine in ax.spines.values():
        spine.set_color('#333333')
    
    # 设置刻度标签格式
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('student_correct_rate_distribution.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # 显示图形
    plt.show()

def main():
    file_numbers = range(1, 11)
    correct_rates = load_and_process_data(file_numbers)
    plot_normal_distribution(correct_rates)

if __name__ == "__main__":
    main() 