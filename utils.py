import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, mean_squared_error

def calculate_student_performance(sequences):
    """计算每个学生的平均正确率"""
    student_performance = {}
    
    for seq in sequences:
        student_id = seq['student_id']
        responses = seq['response_seq']
        valid_responses = [r for r in responses if r != -1]
        if valid_responses:
            accuracy = sum(valid_responses) / len(valid_responses)
            student_performance[student_id] = accuracy
            
    return student_performance

def classify_students(student_performance):
    """根据正确率将学生分类"""
    student_categories = {}
    
    for student_id, accuracy in student_performance.items():
        if accuracy > 0.518:
            student_categories[student_id] = 'good'
        elif accuracy > 0.280:
            student_categories[student_id] = 'medium'
        else:
            student_categories[student_id] = 'poor'
            
    return student_categories

def get_gate_weights(student_category):
    """根据学生类别返回学习门和遗忘门的权重"""
    weights = {
        'good': (1.2, 0.8),
        'medium': (1.0, 1.0),
        'poor': (0.8, 1.2)
    }
    return weights.get(student_category, (1.0, 1.0))

def calculate_skill_weights(training_data_path, num_skills):
    """计算知识点权重"""
    try:
        df = pd.read_csv(training_data_path)
    except FileNotFoundError:
        df = pd.read_csv('student_log_12.csv')
        # 确保有skill_id列，如果没有则处理
        if 'skill_id' not in df.columns and 'skill' in df.columns:
            print("Creating skill_id from skill column")
            skill_map = {skill: idx for idx, skill in enumerate(sorted(df['skill'].unique()))}
            df['skill_id'] = df['skill'].map(skill_map)
    except Exception as e:
        print(f"Warning: Could not calculate skill weights: {e}")
        return torch.ones(num_skills)
    
    # 确保skill_id列存在
    if 'skill_id' not in df.columns:
        print("Warning: Could not calculate skill weights: 'skill_id' column not found")
        return torch.ones(num_skills)
    
    # 确保skill_id是整数类型
    df['skill_id'] = df['skill_id'].astype(int)
    
    skill_counts = df['skill_id'].value_counts()
    total_count = skill_counts.sum()
    
    skill_weights = skill_counts / total_count
    weights_tensor = torch.ones(num_skills)
    
    for skill_id, weight in skill_weights.items():
        if 0 <= skill_id < num_skills:
            weights_tensor[skill_id] = weight
    
    # 打印调试信息
    print(f"Skill weights calculated for {len(skill_weights)} skills")
    print(f"Weights range: [{weights_tensor.min().item():.4f}, {weights_tensor.max().item():.4f}]")
    
    return weights_tensor

def calculate_metrics(y_true, y_pred):
    """计算模型性能指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测值（原始预测概率）
        
    Returns:
        dict: 包含AUC、ACC、RMSE的字典
    """
    # 确保输入是numpy数组
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 移除无效值（-1）
    valid_mask = y_true != -1
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    # 确保预测值在[0,1]范围内
    y_pred = np.clip(y_pred, 1e-6, 1-1e-6)
    
    # 将真实值转换为整数类型的二元标签
    y_true = (y_true > 0.5).astype(np.int32)
    
    # 检查是否有足够的样本和类别
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        print("Warning: Not enough samples or unique classes for metric calculation")
        return {
            'AUC': 0.5,
            'ACC': float(np.mean(y_true == (y_pred > 0.5))),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred)))
        }
    
    try:
        # 计算AUC（使用原始预测概率）
        auc = roc_auc_score(y_true, y_pred)
        
        # 计算ACC（使用二值化的预测）
        y_pred_binary = (y_pred >= 0.5).astype(np.int32)
        acc = accuracy_score(y_true, y_pred_binary)
        
        # 计算RMSE（使用原始预测概率）
        rmse = np.sqrt(mean_squared_error(y_true.astype(np.float32), y_pred))
        
        # 打印调试信息
        print(f"Debug - Metrics calculation:")
        print(f"Number of samples: {len(y_true)}")
        print(f"Unique true values: {np.unique(y_true, return_counts=True)}")
        print(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        print(f"Calculated metrics - AUC: {auc:.4f}, ACC: {acc:.4f}, RMSE: {rmse:.4f}")
        
        return {
            'AUC': float(auc),
            'ACC': float(acc),
            'RMSE': float(rmse)
        }
    except Exception as e:
        print(f"Error in metric calculation: {e}")
        print(f"y_true shape: {y_true.shape}, unique values: {np.unique(y_true, return_counts=True)}")
        print(f"y_pred shape: {y_pred.shape}, range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        
        # 尝试使用备选方法计算指标
        try:
            acc = np.mean(y_true == (y_pred >= 0.5).astype(np.int32))
            rmse = np.sqrt(np.mean((y_true.astype(np.float32) - y_pred) ** 2))
            return {
                'AUC': 0.5,
                'ACC': float(acc),
                'RMSE': float(rmse)
            }
        except:
            return {
                'AUC': 0.5,
                'ACC': 0.0,
                'RMSE': float('inf')
            } 