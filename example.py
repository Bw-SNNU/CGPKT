from models import EKMFKT
from pretrain import PretrainEmbedding
import pandas as pd
import time
import torch
import numpy as np

def load_data(data_path, q_matrix_path):
    """加载和预处理数据
    
    处理：
    1. 学生答题记录
    2. Q矩阵（知识点-试题关系矩阵）
    3. 特征提取和标准化
    """
    # 读取学生答题记录
    df = pd.read_csv(data_path)
    
    # 读取Q矩阵
    q_matrix = np.loadtxt(q_matrix_path, delimiter=',')
    
    # 按学生ID分组，获取每个学生的答题序列
    sequences = []
    for student_id, group in df.groupby('student_id'):
        seq = {
            'exercise_seq': group['exercise_id'].values.astype(np.int64),
            'skill_seq': group['skill_id'].values.astype(np.int64),
            'response_seq': np.clip(group['correct'].values, 0, 1).astype(np.float32),  # 确保在[0,1]范围
            'time_seq': group['response_time'].values.astype(np.float32),
            'interval_seq': np.zeros(len(group), dtype=np.float32),  # 时间间隔为0
            'attempt_seq': group['attempt_count'].values.astype(np.float32),
            'hint_seq': group['hint_count'].values.astype(np.float32)
        }
        sequences.append(seq)
    
    return sequences, q_matrix, len(df['exercise_id'].unique()), len(df['skill_id'].unique())

def prepare_batch(sequences, batch_size, device):
    """准备训练批次数据
    
    处理：
    1. 序列填充到相同长度
    2. 数据类型转换
    3. 设备迁移
    4. 掩码生成
    """
    batch_indices = np.random.choice(len(sequences), batch_size)
    batch_seqs = [sequences[i] for i in batch_indices]
    
    max_len = max(len(seq['exercise_seq']) for seq in batch_seqs)
    
    batch = {
        'exercise_seq': torch.zeros(batch_size, max_len, dtype=torch.long),
        'skill_seq': torch.zeros(batch_size, max_len, dtype=torch.long),
        'response_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'time_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'interval_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'attempt_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'hint_seq': torch.zeros(batch_size, max_len, dtype=torch.float),
        'mask_seq': torch.zeros(batch_size, max_len, dtype=torch.float)
    }
    
    for i, seq in enumerate(batch_seqs):
        length = len(seq['exercise_seq'])
        if length > 0:
            batch['exercise_seq'][i, :length] = torch.tensor(seq['exercise_seq'], dtype=torch.long)
            batch['skill_seq'][i, :length] = torch.tensor(seq['skill_seq'], dtype=torch.long)
            batch['response_seq'][i, :length] = torch.tensor(seq['response_seq'], dtype=torch.float).clamp(0, 1)
            batch['time_seq'][i, :length] = torch.tensor(seq['time_seq'], dtype=torch.float)
            batch['interval_seq'][i, :length] = torch.tensor(seq['interval_seq'], dtype=torch.float)
            batch['attempt_seq'][i, :length] = torch.tensor(seq['attempt_seq'], dtype=torch.float)
            batch['hint_seq'][i, :length] = torch.tensor(seq['hint_seq'], dtype=torch.float)
            batch['mask_seq'][i, :length] = 1.0
    
    return {k: v.to(device) for k, v in batch.items()}

def normalize_feature(feature):
    """特征归一化
    
    将特征值归一化到[0,1]范围，处理边界情况
    """
    if len(feature) == 0:
        return feature
    min_val = feature.min()
    max_val = feature.max()
    if max_val == min_val:
        return np.zeros_like(feature)
    return (feature - min_val) / (max_val - min_val)

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 修改为使用CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 修改为当前目录下的文件路径
    data_path = 'training_data.csv'  # 学生答题记录
    q_matrix_path = 'q_matrix.csv'   # Q矩阵文件
    
    # 加载数据
    sequences, q_matrix, num_exercises, num_skills = load_data(data_path, q_matrix_path)
    
    # 将q_matrix转换为tensor并移到正确的设备上
    q_matrix = torch.tensor(q_matrix, dtype=torch.float).to(device)
    
    # 模型参数
    embed_dim = 128     # 嵌入维度改为128
    hidden_dim = 128    # 隐藏层维度改为128
    batch_size = 32     # 批次大小
    
    # 初始化预训练模型
    pretrain_model = PretrainEmbedding(
        num_exercises=num_exercises,
        num_skills=num_skills,
        embed_dim=embed_dim,
        q_matrix=q_matrix  # 已经是tensor并在正确的设备上
    ).to(device)
    
    # 预训练参数
    num_epochs = 50         # 预训练轮数
    learning_rate = 0.001   # 学习率
    
    # 优化器
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 每5个epoch衰减学习率
    
    # 预训练过程
    print("开始预训练...")
    for epoch in range(num_epochs):
        # 生成训练对
        pairs = pretrain_model.generate_pairs()
        # 确保pairs是long类型的tensor并且在正确的设备上
        pairs = torch.tensor(pairs, dtype=torch.long, device=device)
        
        # 训练一步
        loss = pretrain_model.train_step(optimizer, pairs)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")
        
        scheduler.step()  # 更新学习率
    
    # 获取预训练的嵌入
    exercise_embeddings, skill_embeddings = pretrain_model.get_embeddings()
    print("\n预训练完成！")
    
    # 初始化主模型
    model = EKMFKT(
        num_exercises=num_exercises,
        num_skills=num_skills,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim
    ).to(device)
    
    # 使用预训练的嵌入初始化主模型的嵌入层
    with torch.no_grad():
        model.exercise_embed.weight.copy_(exercise_embeddings)
        model.skill_embed.weight.copy_(skill_embeddings)
    
    print("模型初始化完成！")
    
    # 数据划分
    indices = np.random.permutation(len(sequences))
    train_size = int(0.8 * len(sequences))
    val_size = int(0.1 * len(sequences))

    train_sequences = [sequences[i] for i in indices[:train_size]]
    val_sequences = [sequences[i] for i in indices[train_size:train_size+val_size]]
    test_sequences = [sequences[i] for i in indices[train_size+val_size:]]

    # 训练循环
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = len(train_sequences) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = prepare_batch(train_sequences[start_idx:end_idx], batch_size, device)
            
            # 前向传播
            pred_seq = model(
                exercise_seq=batch['exercise_seq'],
                skill_seq=batch['skill_seq'],
                response_seq=batch['response_seq'],
                time_seq=batch['time_seq'],
                interval_seq=batch['interval_seq'],
                attempt_seq=batch['attempt_seq'],
                hint_seq=batch['hint_seq'],
                q_matrix=q_matrix
            )
            
            # 计算损失 - 确保response_seq在[0,1]范围内
            loss = model.loss(
                pred_seq, 
                torch.clamp(batch['response_seq'], 0, 1),  # 确保目标值在[0,1]范围内
                batch['mask_seq']
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
        
        scheduler.step()  # 更新学习率

    print("\nEvaluating model...")
    # 评估模式
    model.eval()
    with torch.no_grad():  # 不计算梯度
        # 使用训练数据进行测试
        pred_seq = model(
            exercise_seq=batch['exercise_seq'],
            skill_seq=batch['skill_seq'],
            response_seq=batch['response_seq'],
            time_seq=batch['time_seq'],
            interval_seq=batch['interval_seq'],
            attempt_seq=batch['attempt_seq'],
            hint_seq=batch['hint_seq'],
            q_matrix=q_matrix
        )
        # 计算测试损失
        test_loss = model.loss(pred_seq, batch['response_seq'], batch['mask_seq'])
        print(f"Final Test Loss: {test_loss.item():.4f}")

    print("\nSaving model...")
    # 保存训练好的模型参数
    torch.save(model.state_dict(), 'ekmfkt_model.pth')
    print("Model saved to ekmfkt_model.pth")

def evaluate(model, sequences, batch_size, device, q_matrix):
    """模型评估函数
    
    在给定数据集上评估模型性能，计算平均损失
    """
    model.eval()
    total_loss = 0
    num_batches = len(sequences) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = prepare_batch(sequences[start_idx:end_idx], batch_size, device)
            
            pred_seq = model(
                exercise_seq=batch['exercise_seq'],
                skill_seq=batch['skill_seq'],
                response_seq=batch['response_seq'],
                time_seq=batch['time_seq'],
                interval_seq=batch['interval_seq'],
                attempt_seq=batch['attempt_seq'],
                hint_seq=batch['hint_seq'],
                q_matrix=q_matrix
            )
            
            loss = model.loss(pred_seq, batch['response_seq'], batch['mask_seq'])
            total_loss += loss.item()
    
    return total_loss / num_batches

if __name__ == "__main__":
    main() 