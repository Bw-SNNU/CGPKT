import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split


def process_data(input_file, max_seq_len=20):
    """
    处理student_log数据
    
    参数:
    - input_file: 输入文件路径
    - max_seq_len: 最大序列长度，默认100
    """
    # 读取原始数据
    df = pd.read_csv(input_file)
    
    # 1. 规范化response_time（使用log变换和z-score标准化）
    df['response_time'] = df['response_time'].astype(float)
    df['response_time'] = np.log1p(df['response_time'])  # log1p处理偏态分布
    df['response_time'] = (df['response_time'] - df['response_time'].mean()) / df['response_time'].std()  # z-score标准化
    
    # 2. 处理time_interval
    df['time_interval'] = df['time_interval'].fillna(0)
    # 对非零的time_interval进行log变换和标准化
    mask = df['time_interval'] > 0
    if mask.any():
        df.loc[mask, 'time_interval'] = np.log1p(df.loc[mask, 'time_interval'])
        mean_interval = df.loc[mask, 'time_interval'].mean()
        std_interval = df.loc[mask, 'time_interval'].std()
        if std_interval > 0:
            df.loc[mask, 'time_interval'] = (df.loc[mask, 'time_interval'] - mean_interval) / std_interval
    
    # 3. 对于同一题目的多次尝试，只保留最后一次记录（可能更能反映学生的最终掌握程度）
    df = df.sort_values(['student_id', 'exercise_id', 'response_time'])
    df = df.drop_duplicates(subset=['student_id', 'exercise_id'], keep='last')
    
    # 4. 按照时间顺序对每个学生的答题记录进行排序
    df = df.sort_values(['student_id', 'response_time'])
    
    # 5. 知识点映射，确保从0开始
    skill_map = {skill: idx for idx, skill in enumerate(sorted(df['skill'].unique()))}
    df['skill_id'] = df['skill'].map(skill_map)
    
    # 6. 试题ID映射，确保从0开始
    exercise_map = {ex: idx for idx, ex in enumerate(sorted(df['exercise_id'].unique()))}
    df['exercise_id_mapped'] = df['exercise_id'].map(exercise_map)
    
    # 7. 生成Q矩阵
    num_exercises = len(exercise_map)
    num_skills = len(skill_map)
    q_matrix = np.zeros((num_exercises, num_skills))
    
    # 填充Q矩阵
    for _, row in df.iterrows():
        q_matrix[int(row['exercise_id_mapped']), int(row['skill_id'])] = 1
    
    # 8. 按学生分组并处理序列长度
    sequences = []
    for student_id, group in df.groupby('student_id'):
        if len(group) > max_seq_len:
            group = group.iloc[:max_seq_len]
        
        # 确保所有数值特征在合理范围内
        seq = {
            'exercise_seq': group['exercise_id_mapped'].values.astype(np.int64),
            'skill_seq': group['skill_id'].values.astype(np.int64),
            'response_seq': np.clip(group['correct'].values, 0, 1).astype(np.float32),
            'time_seq': group['response_time'].values.astype(np.float32),
            'interval_seq': group['time_interval'].values.astype(np.float32),
            'attempt_seq': group['attempt_count'].values.astype(np.float32),
            'hint_seq': group['hint_count'].values.astype(np.float32),
            'mask_seq': np.ones(len(group), dtype=np.float32),
            'student_id': student_id
        }
        sequences.append(seq)
    
    # 打印数据处理统计信息
    print(f"\n数据预处理统计:")
    print(f"原始记录数: {len(df)}")
    print(f"学生数量: {df['student_id'].nunique()}")
    print(f"题目数量: {num_exercises}")
    print(f"知识点数量: {num_skills}")
    print(f"response_time统计:")
    print(f"  - 范围: [{df['response_time'].min():.4f}, {df['response_time'].max():.4f}]")
    print(f"  - 均值: {df['response_time'].mean():.4f}")
    print(f"  - 标准差: {df['response_time'].std():.4f}")
    print(f"time_interval统计:")
    print(f"  - 范围: [{df['time_interval'].min():.4f}, {df['time_interval'].max():.4f}]")
    print(f"  - 均值: {df['time_interval'].mean():.4f}")
    print(f"  - 标准差: {df['time_interval'].std():.4f}")
    print(f"正确率分布: {df['correct'].value_counts(normalize=True).round(4)}")
    
    return sequences, q_matrix, num_exercises, num_skills


# 生成训练数据
sequences, q_matrix, num_exercises, num_skills = process_data('student_log_1.csv', max_seq_len=20)

def create_dataloaders(dataset, batch_size=32):
    """
    创建训练、验证和测试数据加载器
    
    参数:
    - dataset: 完整数据集
    - batch_size: 每个批次的大小，默认32
    
    返回:
    - train_loader: 训练数据加载器
    - valid_loader: 验证数据加载器
    - test_loader: 测试数据加载器
    """
    # 计算数据集划分的大小
    train_size = int(0.8 * len(dataset))  # 80%用于训练
    valid_size = int(0.1 * len(dataset))  # 10%用于验证
    test_size = len(dataset) - train_size - valid_size  # 剩余用于测试
    
    # 随机划分数据集
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader, test_loader