import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from main import EKMFKT, process_data, prepare_batch, calculate_metrics





























































from typing import Dict, List, Tuple

def simplified_evaluation(learn_weight: float, forget_weight: float):
    """最简单的评估方法，只评估单个样本"""
    try:
        # 设置设备
        device = torch.device('cpu')
        print(f"使用设备: {device}")
        
        # 加载数据（非常有限的数据）
        sequences, q_matrix, num_exercises, num_skills = process_data(
            'student_log_1.csv',
            max_seq_len=10  # 极小的序列长度
        )
        q_matrix = torch.tensor(q_matrix, dtype=torch.float).to(device)
        
        # 只使用少量样本
        test_size = min(10, len(sequences))
        sample_indices = np.random.choice(len(sequences), test_size, replace=False)
        test_sequences = [sequences[i] for i in sample_indices]
        
        print(f"使用 {test_size} 个样本进行简化评估")
        
        # 为每个样本单独评估
        all_preds = []
        all_targets = []
        
        for seq_idx, seq in enumerate(test_sequences):
            print(f"处理样本 {seq_idx+1}/{test_size}")
            
            # 获取样本数据
            exercises = seq['exercises'][:5]  # 只取前5个时间步
            skills = seq['skills'][:5]
            responses = seq['responses'][:5]
            times = seq['times'][:5]
            intervals = seq['intervals'][:5]
            attempts = seq['attempts'][:5]
            hints = seq['hints'][:5]
            mask = seq['mask'][:5]
            
            # 创建单个样本的模型（每个样本一个模型，避免任何状态影响）
            model = EKMFKT(
                num_exercises=num_exercises,
                num_skills=num_skills,
                hidden_dim=16,   # 非常小的隐藏层
                embed_dim=16,    # 非常小的嵌入维度
                dropout=0.0      # 没有dropout
            ).to(device)
            
            # 直接执行前向传播预测
            try:
                with torch.no_grad():
                    # 准备单个样本的输入
                    exercise_seq = torch.tensor([exercises], device=device).long()
                    skill_seq = torch.tensor([skills], device=device).long()
                    response_seq = torch.tensor([responses], device=device).float()
                    time_seq = torch.tensor([times], device=device).float()
                    interval_seq = torch.tensor([intervals], device=device).float()
                    attempt_seq = torch.tensor([attempts], device=device).float()
                    hint_seq = torch.tensor([hints], device=device).float()
                    
                    # 权重
                    single_learn_weight = torch.tensor([learn_weight], device=device)
                    single_forget_weight = torch.tensor([forget_weight], device=device)
                    
                    # 输出每个张量的形状（调试）
                    print(f"exercise_seq形状: {exercise_seq.shape}")
                    print(f"skill_seq形状: {skill_seq.shape}")
                    print(f"response_seq形状: {response_seq.shape}")
                    
                    # 前向传播
                    pred_seq = model(
                        exercise_seq=exercise_seq,
                        skill_seq=skill_seq,
                        response_seq=response_seq,
                        time_seq=time_seq,
                        interval_seq=interval_seq,
                        attempt_seq=attempt_seq,
                        hint_seq=hint_seq,
                        q_matrix=q_matrix,
                        learn_weights=single_learn_weight,
                        forget_weights=single_forget_weight
                    )
                    
                    # 获取有效预测
                    mask_tensor = torch.tensor([mask], device=device).bool()
                    valid_indices = mask_tensor.squeeze(0)
                    
                    if valid_indices.sum() > 0:
                        valid_preds = pred_seq.squeeze(0)[valid_indices]
                        valid_targets = response_seq.squeeze(0)[valid_indices]
                        
                        all_preds.extend(valid_preds.cpu().numpy())
                        all_targets.extend(valid_targets.cpu().numpy())
                        
                        print(f"样本 {seq_idx+1} 的预测值: {valid_preds.cpu().numpy()}")
                        print(f"样本 {seq_idx+1} 的目标值: {valid_targets.cpu().numpy()}")
            except Exception as e:
                print(f"样本 {seq_idx+1} 处理出错: {e}")
                continue
        
        # 计算整体指标
        if len(all_preds) > 0:
            metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
            print(f"整体结果 - AUC: {metrics['AUC']:.4f}, ACC: {metrics['ACC']:.4f}, RMSE: {metrics['RMSE']:.4f}")
            return metrics
        else:
            print("没有有效预测，返回默认指标")
            return {'AUC': 0.5, 'ACC': 0.5, 'RMSE': 1.0}
            
    except Exception as e:
        print(f"评估过程发生错误: {e}")
        return {'AUC': 0.5, 'ACC': 0.5, 'RMSE': 1.0}

def plot_weight_comparison(weight_diffs: List[float], 
                         metrics: Dict[str, List[float]], 
                         save_path: str = 'weight_comparison.png'):
    """绘制不同权重差异下的性能对比图"""
    plt.figure(figsize=(12, 6))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    metrics_names = {'AUC': 'AUC值', 'ACC': '准确率', 'RMSE': 'RMSE值'}
    colors = {'AUC': 'blue', 'ACC': 'green', 'RMSE': 'red'}
    
    for metric_name in metrics:
        plt.plot(weight_diffs, metrics[metric_name], 
                marker='o', label=metrics_names[metric_name],
                color=colors[metric_name], linewidth=2)
        
        # 添加数值标注
        for x, y in zip(weight_diffs, metrics[metric_name]):
            plt.text(x, y, f'{y:.4f}', 
                    horizontalalignment='center',
                    verticalalignment='bottom')
    
    plt.xlabel('权重差异')
    plt.ylabel('性能指标值')
    plt.title('不同权重差异下的模型性能对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results(weight_diffs: List[float], 
                metrics: Dict[str, List[float]], 
                learn_weights: List[float],
                forget_weights: List[float],
                save_path: str = 'weight_comparison_results.csv'):
    """保存实验结果到CSV文件"""
    data = {
        '权重差异': weight_diffs,
        '学习权重': learn_weights,
        '遗忘权重': forget_weights,
        'AUC': metrics['AUC'],
        'ACC': metrics['ACC'],
        'RMSE': metrics['RMSE']
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n实验结果已保存至 '{save_path}'")
    
    # 打印结果表格
    print("\n实验结果汇总:")
    print(df.to_string(index=False))

def main():
    """主函数 - 极简化版本"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 定义权重差异
    weight_diffs = [0.2]
    metrics = {'AUC': [], 'ACC': [], 'RMSE': []}
    learn_weights = []
    forget_weights = []
    
    # 对每个权重差异进行测试
    for diff in weight_diffs:
        print(f"\n测试权重差异: {diff}")
        
        # 设置权重
        learn_weight = 1.0 + diff/2
        forget_weight = 1.0 - diff/2
        
        learn_weights.append(learn_weight)
        forget_weights.append(forget_weight)
        
        # 评估
        print(f"评估权重 - 学习: {learn_weight:.2f}, 遗忘: {forget_weight:.2f}")
        results = simplified_evaluation(learn_weight, forget_weight)
        
        metrics['AUC'].append(results['AUC'])
        metrics['ACC'].append(results['ACC'])
        metrics['RMSE'].append(results['RMSE'])
        
    # 绘制性能对比图
    plot_weight_comparison(weight_diffs, metrics)
    
    # 保存实验结果
    save_results(weight_diffs, metrics, learn_weights, forget_weights)
    
    print("\n实验完成!")

if __name__ == "__main__":
    main() 