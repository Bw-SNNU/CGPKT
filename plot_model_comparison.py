import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

# 保持Agg后端以减少警告，但不完全消除
matplotlib.use('Agg')

def plot_model_comparisons():
    # 确保输出目录存在
    output_dir = 'model_comparison_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体和全局样式
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['legend.fontsize'] = 12 # 设置全局图例字号
    
    # 第一组数据 (ASSISTChall)
    models1 = ['AKT', 'LPKT', 'LBKT', 'EKMFKT', 'CPKT', 'GPKT', 'CGPKT']
    auc_scores1 = [0.7536, 0.7771, 0.7927, 0.8099, 0.8675, 0.8787, 0.9015]
    acc_scores1 = [0.7117, 0.7283, 0.7423, 0.7552, 0.8259, 0.8169, 0.8177]
    rmse_scores1 = [0.4342, 0.4237, 0.4171, 0.4071, 0.3833, 0.4298, 0.3741]
    
    # 第二组数据 (ASSIST2012)
    models2 = ['AKT', 'LPKT', 'LBKT', 'EKMFKT', 'CPKT', 'GPKT', 'CGPKT']
    auc_scores2 = [0.7515, 0.7512, 0.7539, 0.7586, 0.8972, 0.9149, 0.9191]
    acc_scores2 = [0.7412, 0.7401, 0.7415, 0.7439, 0.8404, 0.9242, 0.8330]
    rmse_scores2 = [0.4204, 0.4198, 0.4189, 0.4167, 0.4318, 0.3867, 0.3640]
    
    # 准备数据
    metrics = [
        ('AUC', auc_scores1, auc_scores2),
        ('ACC', acc_scores1, acc_scores2),
        ('RMSE', rmse_scores1, rmse_scores2)
    ]
    
    datasets = ['ASSISTChall', 'ASSIST2012']
    # colors = ['#2878B5', '#9AC9DB', '#C82423', '#F8AC8C', '#C1A570', '#7A8C85', '#5C2223']
    # colors = ['#1A5C9E', '#7AB0C7', '#A8171B', '#E89472', '#A88D5D', '#63746E', '#4A1718']
    # 原色调用于柱状图
    colors = ['#1A5C9E', '#7AB0C7', '#A8171B', '#E89472', '#A88D5D', '#63746E', '#4A1718']
    # 更深的色调用于折线图
    line_colors = ['#14477A', '#5C8EA1', '#7D1013', '#BA6D52', '#806A45', '#4B5851', '#361011']

    # 创建六个独立的图形，每个对应一个指标和数据集组合
    for dataset_idx in range(2):
        for metric_idx in range(3):
            dataset = datasets[dataset_idx]
            metric_name, scores1, scores2 = metrics[metric_idx]
            scores = scores1 if dataset_idx == 0 else scores2
            models = models1 if dataset_idx == 0 else models2
            
            # 为每个子图创建一个新的图形
            plt.figure(figsize=(10, 6))
            
            # 创建主坐标轴
            ax = plt.gca()
            
            # 为每个模型绘制不同颜色的线和标记
            for i, model in enumerate(models):
                ax.plot([i], [scores[i]], marker='o', markersize=10, 
                       color=colors[i], label=model)
                
                # 连接所有点
                if i > 0:
                    ax.plot([i-1, i], [scores[i-1], scores[i]], 
                           color=colors[i-1], linewidth=2)
            
            # 设置轴标签和标题
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, fontsize=14)  # 保持水平显示
            ax.set_title(f'{dataset} - {metric_name}', fontsize=16, pad=15)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 动态调整Y轴范围，为图例腾出空间
            ymin = min(scores)
            ymax = max(scores)
            yrange = ymax - ymin
            if metric_name == 'RMSE':
                top_space = 0.8 # RMSE 预留40%空间
            else:
                top_space = 0.8  # 其他指标预留20%空间
            ax.set_ylim(ymin, ymax + yrange * top_space)
            
            # 设置Y轴标签
            ax.set_ylabel(metric_name, fontsize=14)
            
            # 添加直方图效果 - 增加透明度
            # 创建一个共享x轴的新轴(透明背景)
            ax2 = ax.twinx()
            ax2.set_yticks([])  # 隐藏右侧y轴
            
            # 绘制条形图，增加透明度，在折线图下方显示
            bars = ax2.bar(range(len(models)), scores, alpha=0.7, color=colors[:len(models)], zorder=1)
            
            # 设置背景颜色
            ax.set_facecolor('white')
            
            # 图例放在图外右侧，竖直一列
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=1, fontsize=11, framealpha=0.8)
            
            plt.subplots_adjust(top=0.90, right=0.80)
            filename = f'{dataset}_{metric_name}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            print(f"保存图片: {filepath}")
            plt.close()
    
    # 另外创建一个包含所有子图的组合图
    fig = plt.figure(figsize=(18, 15))
    
    # 依次创建6个子图
    for idx, (dataset_idx, metric_idx) in enumerate([(i, j) for i in range(2) for j in range(3)]):
        dataset = datasets[dataset_idx]
        metric_name, scores1, scores2 = metrics[metric_idx]
        scores = scores1 if dataset_idx == 0 else scores2
        models = models1 if dataset_idx == 0 else models2
        
        # 创建当前子图
        ax = plt.subplot(2, 3, idx + 1)
        
        # 为每个模型绘制不同颜色的线和标记
        for i, model in enumerate(models):
            ax.plot([i], [scores[i]], marker='o', markersize=8, 
                   color=colors[i], label=model)
            
            # 连接所有点
            if i > 0:
                ax.plot([i-1, i], [scores[i-1], scores[i]], 
                       color=colors[i-1], linewidth=2)
        
        # 设置轴标签和标题
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=9)  # 保持水平显示
        ax.set_title(f'{dataset} - {metric_name}', fontsize=14, pad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 动态调整Y轴范围，为图例腾出空间
        ymin = min(scores)
        ymax = max(scores)
        yrange = ymax - ymin
        if metric_name == 'RMSE':
            top_space = 0.4  # RMSE 预留40%空间
        else:
            top_space = 0.2  # 其他指标预留20%空间
        ax.set_ylim(ymin, ymax + yrange * top_space)
        
        # 设置左侧Y轴标签
        ax.set_ylabel(metric_name, fontsize=12)
        
        # 添加直方图效果 - 增加透明度
        # 创建一个共享x轴的新轴(透明背景)
        ax2 = ax.twinx()
        ax2.set_yticks([])  # 隐藏右侧y轴
        
        # 绘制条形图，增加透明度，在折线图下方显示
        bars = ax2.bar(range(len(models)), scores, alpha=0.7, color=colors[:len(models)], zorder=1)
        
        # 设置背景颜色
        ax.set_facecolor('white')
        
        # 图例放在图外右侧，竖直一列
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=1, fontsize=8, framealpha=0.8)
    
    plt.subplots_adjust(top=0.90, right=0.80)
    combined_filepath = os.path.join(output_dir, 'model_comparisons_combined.png')
    plt.savefig(combined_filepath, dpi=300)
    print(f"保存组合图片: {combined_filepath}")
    
    # 关闭图形
    plt.close()

if __name__ == "__main__":
    plot_model_comparisons() 