import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple
from data import process_data
from example import prepare_batch, normalize_feature
import copy
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, mean_squared_error
# from models import (EKMFKT, EKMFKT_NoResponseTime, EKMFKT_NoIntervalTime,
#                    EKMFKT_NoBehavior, EKMFKT_NoForgetGate, EKMFKT_NoLearnGate)
# from utils import (calculate_student_performance, classify_students,
#                   get_gate_weights, calculate_metrics)

def calculate_student_performance(sequences):
    """计算每个学生的平均正确率
    
    Args:
        sequences: 学生答题序列数据
        
    Returns:
        dict: student_id到正确率的映射
    """
    student_performance = {}
    
    for seq in sequences:
        student_id = seq['student_id']
        responses = seq['response_seq']
        # 计算非填充值的平均正确率
        valid_responses = [r for r in responses if r != -1]  # 排除填充值
        if valid_responses:
            accuracy = sum(valid_responses) / len(valid_responses)
            student_performance[student_id] = accuracy
            
    return student_performance

def classify_students(student_performance):
    """根据正确率将学生分类
    
    分类标准:
    - 好: 正确率 > 0.518
    - 中: 0.280 < 正确率 ≤ 0.518
    - 差: 正确率 ≤ 0.280
    
    Args:
        student_performance: dict，学生ID到正确率的映射
        
    Returns:
        dict: 学生ID到分类的映射 ('good', 'medium', 'poor')
    """
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
    """根据学生类别返回学习门和遗忘门的权重
    
    Args:
        student_category: str, 'good', 'medium', 或 'poor'
        
    Returns:
        tuple: (学习门权重, 遗忘门权重)
    """
    # weights = {
    #     'good': (1.05, 0.95),    # (学习门权重更高，遗忘门权重更低)
    #     'medium': (1.0, 1.0),  # (保持默认权重)
    #     'poor': (0.95, 1.05)     # (学习门权重更低，遗忘门权重更高)
    # }
    # weights = {
    #     'good': (1.1, 0.9),  # (学习门权重更高，遗忘门权重更低)
    #     'medium': (1.0, 1.0),  # (保持默认权重)
    #     'poor': (0.9, 1.1)  # (学习门权重更低，遗忘门权重更高)
    # }
    weights = {
        'good': (1.15, 0.85),  # (学习门权重更高，遗忘门权重更低)
        'medium': (1.0, 1.0),  # (保持默认权重)
        'poor': (0.85, 1.15)  # (学习门权重更低，遗忘门权重更高)
    }
    # weights = {
    #     'good': (1.2, 0.8),  # (学习门权重更高，遗忘门权重更低)
    #     'medium': (1.0, 1.0),  # (保持默认权重)
    #     'poor': (0.8, 1.2)  # (学习门权重更低，遗忘门权重更高)
    # }
    # weights = {
    #     'good': (1.25, 0.75),  # (学习门权重更高，遗忘门权重更低)
    #     'medium': (1.0, 1.0),  # (保持默认权重)
    #     'poor': (0.75, 1.25)  # (学习门权重更低，遗忘门权重更高)
    # }
    # weights = {
    #     'good': (1.3, 0.7),  # (学习门权重更高，遗忘门权重更低)
    #     'medium': (1.0, 1.0),  # (保持默认权重)
    #     'poor': (0.7, 1.3)  # (学习门权重更低，遗忘门权重更高)
    # }
    # weights = {
    #     'good': (1.35, 0.65),  # (学习门权重更高，遗忘门权重更低)
    #     'medium': (1.0, 1.0),  # (保持默认权重)
    #     'poor': (0.65, 1.35)  # (学习门权重更低，遗忘门权重更高)
    # }
    return weights.get(student_category, (1.0, 1.0))  # 如果类别不存在，返回默认权重

def calculate_skill_weights(training_data_path, num_skills):
    """计算知识点权重
    
    基于训练数据中各知识点出现的频率计算权重。
    对于未出现的知识点，使用平均权重进行填充。
    """
    try:
        # 尝试读取训练数据
        df = pd.read_csv(training_data_path)
    except FileNotFoundError:
        # 如果找不到training_data.csv，直接使用原始数据文件
        df = pd.read_csv('student_log_1.csv')
    except Exception as e:
        print(f"Error reading data file: {e}")
        # 返回默认权重（全1）
        return torch.ones(num_skills)
    
    # 统计每个知识点出现的次数
    skill_counts = df['skill_id'].value_counts()
    total_count = skill_counts.sum()
    
    # 计算权重
    skill_weights = skill_counts / total_count
    
    # 转换为tensor，确保大小为num_skills
    weights_tensor = torch.zeros(num_skills)
    for skill_id, weight in skill_weights.items():
        if skill_id < num_skills:  # 确保skill_id在有效范围内
            weights_tensor[skill_id] = weight
    
    # 对于未出现的知识点，赋予平均权重
    zero_mask = weights_tensor == 0
    if zero_mask.any():
        avg_weight = weights_tensor.sum() / (~zero_mask).sum()
        weights_tensor[zero_mask] = avg_weight
    
    return weights_tensor

class WeightedLearningGate(nn.Module):
    """
    带权重的学习门模块
    """
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # 定义输入维度 (所有输入张量维度之和)
        # knowledge_state: [batch_size, hidden_dim]
        # interval: [batch_size, 1]
        # curr_interaction: [batch_size, embed_dim]
        # behavior: [batch_size, hidden_dim]
        # skill_weights: [batch_size, 1]
        input_dim = hidden_dim + 1 + embed_dim + hidden_dim + 1
        self.transform = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, knowledge_state, interval, curr_interaction, _, behavior, skill_weights):
        """
        参数:
        - knowledge_state: 知识状态 [batch_size, hidden_dim]
        - interval: 间隔时间 [batch_size]
        - curr_interaction: 当前交互 [batch_size, embed_dim]
        - _: 未使用的参数，原先是prev_interaction
        - behavior: 行为特征 [batch_size, hidden_dim]
        - skill_weights: 知识点权重 [batch_size]
        """
        # 确保维度正确
        interval = interval.unsqueeze(-1)  # [batch_size, 1]
        skill_weights = skill_weights.unsqueeze(-1)  # [batch_size, 1]
        
        # 拼接所有输入
        gate_input = torch.cat([
            knowledge_state,    # [batch_size, hidden_dim]
            interval,           # [batch_size, 1]
            curr_interaction,   # [batch_size, embed_dim]
            behavior,           # [batch_size, hidden_dim]
            skill_weights       # [batch_size, 1]
        ], dim=-1)
        
        # 应用线性变换和sigmoid激活
        return torch.sigmoid(self.transform(gate_input))

class WeightedForgetGate(nn.Module):
    """
    带权重的遗忘门模块
    """
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # 定义输入维度 (所有输入张量维度之和)
        # knowledge_state: [batch_size, hidden_dim]
        # curr_interaction: [batch_size, embed_dim]
        # learning_gain: [batch_size, hidden_dim]
        # behavior: [batch_size, hidden_dim]
        # skill_weights: [batch_size, 1]
        input_dim = hidden_dim + embed_dim + hidden_dim + hidden_dim + 1
        self.transform = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, knowledge_state, curr_interaction, learning_gain, behavior, skill_weights):
        """
        参数:
        - knowledge_state: 知识状态 [batch_size, hidden_dim]
        - curr_interaction: 当前交互 [batch_size, embed_dim]
        - learning_gain: 学习增益 [batch_size, hidden_dim]
        - behavior: 行为特征 [batch_size, hidden_dim]
        - skill_weights: 知识点权重 [batch_size]
        """
        # 确保维度正确
        skill_weights = skill_weights.unsqueeze(-1)  # [batch_size, 1]
        
        # 拼接所有输入
        gate_input = torch.cat([
            knowledge_state,    # [batch_size, hidden_dim]
            curr_interaction,   # [batch_size, embed_dim]
            learning_gain,      # [batch_size, hidden_dim]
            behavior,           # [batch_size, hidden_dim]
            skill_weights       # [batch_size, 1]
        ], dim=-1)
        
        # 应用线性变换和sigmoid激活
        return torch.sigmoid(self.transform(gate_input))

class EKMFKT(nn.Module):
    def __init__(self, 
                 num_exercises: int,  # 试题总数
                 num_skills: int,     # 知识点总数
                 hidden_dim: int = 128,  # 隐藏层维度
                 embed_dim: int = 128,   # 嵌入维度
                 dropout: float = 0.2):  # dropout比率
        """
        模型初始化
        
        参数说明:
        - num_exercises: 数据集中试题的总数量
        - num_skills: 数据集中知识点的总数量
        - hidden_dim: 模型隐藏层的维度，用于知识状态表示
        - embed_dim: 嵌入层的维度，用于特征表示
        - dropout: dropout比率，用于防止过拟合
        """
        super().__init__()
        
        self.num_exercises = num_exercises
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # 计算知识点权重
        try:
            self.skill_weights = calculate_skill_weights('./training_data.csv', num_skills)
        except Exception as e:
            print(f"Warning: Could not calculate skill weights: {e}")
            print("Using uniform weights instead.")
            self.skill_weights = torch.ones(num_skills)
        
        # 嵌入层
        self.exercise_embed = nn.Embedding(num_exercises, embed_dim)  # e_t
        self.skill_embed = nn.Embedding(num_skills, embed_dim)      # s_t
        self.response_embed = nn.Embedding(2, embed_dim)           # r_t
        
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.exercise_embed.weight)
        nn.init.xavier_uniform_(self.skill_embed.weight)
        nn.init.xavier_uniform_(self.response_embed.weight)
        
        # 时间特征转换层
        self.time_transform = nn.Linear(1, embed_dim)  # at_t
        
        # 交互嵌入转换层 W_1[e_t ⊕ s_t ⊕ at_t ⊕ r_t] + b_1
        self.interaction_transform = nn.Linear(embed_dim * 4, embed_dim)
        
        # 行为特征转换层
        self.behavior_transform = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 2个特征：attempt, hint
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 带权重的学习门和遗忘门
        self.learning_gate = WeightedLearningGate(hidden_dim, embed_dim)
        self.forget_gate = WeightedForgetGate(hidden_dim, embed_dim)
        
        # 知识状态转换
        total_knowledge_dim = hidden_dim + embed_dim + 1 + hidden_dim + hidden_dim
        self.knowledge_transform = nn.Linear(total_knowledge_dim, hidden_dim)
        
        # 预测模块 - 修改输入维度，确保匹配
        self.predictor = nn.Linear(hidden_dim + embed_dim, 1)
        
        # 正则化和激活函数
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # 添加权重调整参数
        self.weight_alpha = nn.Parameter(torch.tensor(1.0))  # 补偿系数
        self.weight_beta = nn.Parameter(torch.tensor(1.0))   # 缩放因子
        
        # 确保不同变体模型使用相同的predictor层维度
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
    
    def compute_interaction_embedding(self, exercise_embed, skill_embed, response, time):
        """
        计算交互嵌入: i_t = LeakyReLU(W_1[e_t ⊕ s_t ⊕ at_t ⊕ r_t] + b_1)
        
        参数:
        - exercise_embed: 试题嵌入 e_t [batch_size, embed_dim]
        - skill_embed: 知识点嵌入 s_t [batch_size, embed_dim]
        - response: 作答结果 [batch_size]
        - time: 响应时间 [batch_size]
        """
        # 获取作答结果嵌入 r_t
        response_embed = self.response_embed(response.long())
        
        # 转换响应时间 at_t，确保维度正确
        time = time.float().view(-1, 1)  # 将time转换为[batch_size, 1]
        time_embed = self.time_transform(time)
        
        # 拼接四个向量 [e_t ⊕ s_t ⊕ at_t ⊕ r_t]
        interaction = torch.cat([
            exercise_embed,  # e_t：试题嵌入
            skill_embed,    # s_t：知识点嵌入
            time_embed,     # at_t：响应时间
            response_embed  # r_t：作答结果
        ], dim=-1)
        
        # 应用线性变换和LeakyReLU激活
        return F.leaky_relu(self.interaction_transform(interaction))
    
    def knowledge_update(self, knowledge_state, curr_interaction, prev_interaction, interval, behavior, skill_ids):
        """
        更新知识状态
        
        参数:
        - knowledge_state: 当前知识状态 [batch_size, hidden_dim]
        - curr_interaction: 当前交互 [batch_size, embed_dim]
        - prev_interaction: 前一个交互，实际上可能没有使用 [batch_size, embed_dim]
        - interval: 时间间隔 [batch_size]
        - behavior: 行为特征 [batch_size, hidden_dim]
        - skill_ids: 知识点ID [batch_size]
        
        返回:
        - 更新后的知识状态 [batch_size, hidden_dim]
        """
        # 获取知识点权重
        batch_skill_weights = self.skill_weights[skill_ids]
        
        # 计算学习增益
        learning_gain = self.learning_gate(
            knowledge_state,      # 知识状态
            interval,             # 时间间隔
            curr_interaction,     # 当前交互
            knowledge_state,      # 占位参数，实际未使用
            behavior,             # 行为特征
            batch_skill_weights   # 知识点权重
        )
        learning_gain = l_w * learning_gain  # 应用学生特定的学习权重
        
        # 计算遗忘门输出
        forget_gate = self.forget_gate(
            knowledge_state,      # 知识状态
            curr_interaction,     # 当前交互
            learning_gain,        # 学习增益
            behavior,             # 行为特征
            batch_skill_weights   # 知识点权重
        )
        forget_gate = f_w * forget_gate  # 应用学生特定的遗忘权重
        
        # 更新知识状态: h_t = learning_gain + forget_gate * h_t-1
        knowledge_state = learning_gain + forget_gate * knowledge_state
        
        return knowledge_state
    
    def forward(self, 
                exercise_seq: torch.Tensor,   
                skill_seq: torch.Tensor,      
                response_seq: torch.Tensor,   
                time_seq: torch.Tensor,       
                interval_seq: torch.Tensor,   
                attempt_seq: torch.Tensor,    
                hint_seq: torch.Tensor,       
                q_matrix: torch.Tensor,       
                learn_weights=None,           
                forget_weights=None           
                ) -> torch.Tensor:
        """
        前向传播，增加了基于学生表现的门控权重
        """
        # 确保输入的类型正确
        exercise_seq = exercise_seq.long()
        skill_seq = skill_seq.long()
        time_seq = time_seq.float()  # 确保时间序列是float类型
        
        # 获取嵌入
        exercise_embeds = self.exercise_embed(exercise_seq)
        skill_embeds = self.skill_embed(skill_seq)
        
        # 初始化知识状态
        batch_size = exercise_seq.size(0)
        knowledge_state = torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device)
        
        # 按时间步处理序列
        pred_list = []
        for t in range(exercise_seq.size(1)):
            curr_exercise = exercise_embeds[:, t]
            curr_skill = skill_seq[:, t]  # 获取当前批次的技能ID
            curr_time = time_seq[:, t]
            curr_interval = interval_seq[:, t]
            
            # 计算当前时刻的交互嵌入
            curr_interaction = self.compute_interaction_embedding(
                curr_exercise,
                skill_embeds[:, t],  # 使用已经嵌入的技能向量
                response_seq[:, t],
                curr_time
            )
            
            # 计算行为特征
            curr_behavior = self.behavior_transform(
                torch.stack([attempt_seq[:, t], hint_seq[:, t]], dim=-1)
            )
            
            # 更新知识状态时应用学生特定的权重
            if learn_weights is not None and forget_weights is not None:
                # 扩展权重维度以匹配知识状态维度
                l_w = learn_weights.view(-1, 1)  # [batch_size, 1]
                f_w = forget_weights.view(-1, 1)  # [batch_size, 1]
                
                # 获取当前批次的知识点权重
                batch_skill_weights = self.skill_weights[curr_skill.long()]  # 确保使用long类型的索引
                
                # 计算学习增益
                learning_gain = self.learning_gate(
                    knowledge_state,      # 知识状态
                    curr_interval,        # 时间间隔
                    curr_interaction,     # 当前交互
                    knowledge_state,      # 占位参数，实际未使用
                    curr_behavior,        # 行为特征
                    batch_skill_weights   # 知识点权重
                )
                learning_gain = l_w * learning_gain  # 应用学生特定的学习权重
                
                # 计算遗忘门输出
                forget_gate = self.forget_gate(
                    knowledge_state,      # 知识状态
                    curr_interaction,     # 当前交互
                    learning_gain,        # 学习增益
                    curr_behavior,        # 行为特征
                    batch_skill_weights   # 知识点权重
                )
                forget_gate = f_w * forget_gate  # 应用学生特定的遗忘权重
                
                # 更新知识状态: H_t = LG_t + Γ_t · H_t-1
                knowledge_state = learning_gain + forget_gate * knowledge_state
            else:
            knowledge_state = self.knowledge_update(
                    prev_knowledge=knowledge_state,
                    curr_interaction=curr_interaction,
                    prev_interaction=knowledge_state,
                    interval=curr_interval,
                    behavior=curr_behavior,
                    skill_ids=curr_skill.long()  # 确保使用long类型的索引
                )
            
            # 获取当前批次的知识点权重
            skill_weights = self.skill_weights[curr_skill.long()]  # [batch_size, 1]
            
            # 预测下一步
            # 1. 连接知识状态和当前试题嵌入
            pred_input = torch.cat([
                knowledge_state,      # h_t：当前知识状态 [batch_size, hidden_dim]
                curr_exercise,        # e_t+1：下一个试题嵌入 [batch_size, embed_dim]
            ], dim=-1)
            
            # 2. 生成原始预测值
            raw_pred = self.predictor(pred_input).squeeze(-1)  # [batch_size]
            
            # 3. 获取学生类别权重和知识点权重
            if learn_weights is not None:
                student_weight = learn_weights.view(-1)  # w_c：学生类别权重
            else:
                student_weight = torch.ones_like(raw_pred)
            
            # 4. 权重补偿（按照图二的方式）
            # y_t+1 = σ(W_p[h_t ⊕ e_t+1] + b_p - log(w_c)) · w_c
            compensated_pred = torch.sigmoid(
                raw_pred - torch.log(student_weight + 1e-6)
            ) * student_weight
            
            # 5. 应用知识点权重
            weighted_pred = compensated_pred * skill_weights
            
            # 6. 归一化（按照图三的方式，同时考虑知识点权重）
            # y_normalized = clamp(y_weighted/w_c, 0.05, 0.95)
            normalized_pred = torch.clamp(
                weighted_pred / (student_weight * skill_weights + 1e-6),
                min=0.05,
                max=0.95
            )
            
            pred_list.append(normalized_pred)
        
        # 合并所有时间步的预测结果
        pred_seq = torch.stack(pred_list, dim=1)
        
        return pred_seq

    def loss(self, pred_seq, response_seq, mask_seq):
        """计算二元交叉熵损失，只考虑有效预测"""
        mask = mask_seq.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_seq.device)
            
        valid_preds = pred_seq[mask]
        valid_targets = response_seq[mask]
        
        # 使用二元交叉熵损失
        loss = F.binary_cross_entropy(valid_preds, valid_targets)
        return loss

# 原始模型保持不变，添加以下变体模型

class EKMFKT_NoResponseTime(EKMFKT):
    """不考虑响应时间的变体"""
    def compute_interaction_embedding(self, exercise_embed, skill_embed, response, time):
        """
        计算交互嵌入，不考虑响应时间
        """
        # 将exercise_embed和skill_embed连接
        interaction = torch.cat([exercise_embed, skill_embed], dim=-1)
        
        # 添加response信息
        response = response.unsqueeze(-1)
        interaction = torch.cat([interaction, response], dim=-1)
        
        # 通过全连接层转换
        interaction = self.interaction_transform(interaction)
        
        return interaction

class EKMFKT_NoIntervalTime(EKMFKT):
    """不考虑间隔时间的变体"""
    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        """重写前向传播方法，忽略间隔时间"""
        # 确保输入的类型正确
        exercise_seq = exercise_seq.long()
        skill_seq = skill_seq.long()
        time_seq = time_seq.float()
        
        # 获取嵌入
        exercise_embeds = self.exercise_embed(exercise_seq)
        skill_embeds = self.skill_embed(skill_seq)
        
        # 初始化知识状态
        batch_size = exercise_seq.size(0)
        knowledge_state = torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device)
        pred_list = []
        
        for t in range(exercise_seq.size(1)):
            curr_exercise = exercise_embeds[:, t]
            curr_skill = skill_seq[:, t]
            curr_time = time_seq[:, t]
            
            # 计算当前时刻的交互嵌入
            curr_interaction = self.compute_interaction_embedding(
                curr_exercise,
                skill_embeds[:, t],
                response_seq[:, t],
                curr_time
            )
            
            # 计算行为特征
            curr_behavior = self.behavior_transform(
                torch.stack([attempt_seq[:, t], hint_seq[:, t]], dim=-1)
            )
            
            # 更新知识状态时应用学生特定的权重
            if learn_weights is not None and forget_weights is not None:
                l_w = learn_weights.view(-1, 1)
                f_w = forget_weights.view(-1, 1)
                
                batch_skill_weights = self.skill_weights[curr_skill.long()]
                
                # 使用零间隔时间
                zero_interval = torch.zeros_like(interval_seq[:, t])
                
                learning_gain = self.learning_gate(
                    knowledge_state,      # 知识状态
                    zero_interval,        # 使用零间隔
                    curr_interaction,     # 当前交互
                    knowledge_state,      # 占位参数，实际未使用
                    curr_behavior,        # 行为特征
                    batch_skill_weights   # 知识点权重
                )
                learning_gain = l_w * learning_gain
                
                forget_gate = self.forget_gate(
                    knowledge_state,      # 知识状态
                    curr_interaction,     # 当前交互
                    learning_gain,        # 学习增益
                    curr_behavior,        # 行为特征
                    batch_skill_weights   # 知识点权重
                )
                forget_gate = f_w * forget_gate
                
                knowledge_state = learning_gain + forget_gate * knowledge_state
            else:
            knowledge_state = self.knowledge_update(
                    prev_knowledge=knowledge_state,
                    curr_interaction=curr_interaction,
                    prev_interaction=knowledge_state,
                    interval=torch.zeros_like(interval_seq[:, t]),  # 使用零间隔
                    behavior=curr_behavior,
                    skill_ids=curr_skill.long()
                )
            
            # 获取当前批次的知识点权重
            skill_weights = self.skill_weights[curr_skill.long()]
            
            # 预测下一步
            pred_input = torch.cat([knowledge_state, curr_exercise], dim=-1)
            raw_pred = self.predictor(pred_input).squeeze(-1)
            
            # 应用权重和归一化
            if learn_weights is not None:
                student_weight = learn_weights.view(-1)
            else:
                student_weight = torch.ones_like(raw_pred)
            
            compensated_pred = torch.sigmoid(
                raw_pred - torch.log(student_weight + 1e-6)
            ) * student_weight
            
            weighted_pred = compensated_pred * skill_weights
            normalized_pred = torch.clamp(
                weighted_pred / (student_weight * skill_weights + 1e-6),
                min=0.05,
                max=0.95
            )
            
            pred_list.append(normalized_pred)
        
        pred_seq = torch.stack(pred_list, dim=1)
        return pred_seq

class EKMFKT_NoBehavior(EKMFKT):
    """不考虑行为特征的变体"""
    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        """重写前向传播方法，忽略行为特征"""
        # 确保输入的类型正确
        exercise_seq = exercise_seq.long()
        skill_seq = skill_seq.long()
        time_seq = time_seq.float()
        
        # 获取嵌入
        exercise_embeds = self.exercise_embed(exercise_seq)
        skill_embeds = self.skill_embed(skill_seq)
        
        # 初始化知识状态
        batch_size = exercise_seq.size(0)
        knowledge_state = torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device)
        pred_list = []
        
        # 创建零行为特征
        zero_behavior = torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device)
        
        for t in range(exercise_seq.size(1)):
            curr_exercise = exercise_embeds[:, t]
            curr_skill = skill_seq[:, t]
            curr_time = time_seq[:, t]
            curr_interval = interval_seq[:, t]
            
            # 计算当前时刻的交互嵌入
            curr_interaction = self.compute_interaction_embedding(
                curr_exercise,
                skill_embeds[:, t],
                response_seq[:, t],
                curr_time
            )
            
            # 更新知识状态时应用学生特定的权重
            if learn_weights is not None and forget_weights is not None:
                l_w = learn_weights.view(-1, 1)
                f_w = forget_weights.view(-1, 1)
                
                batch_skill_weights = self.skill_weights[curr_skill.long()]
                
                learning_gain = self.learning_gate(
                    knowledge_state,      # 知识状态
                    curr_interval,        # 时间间隔
                    curr_interaction,     # 当前交互
                    knowledge_state,      # 占位参数，实际未使用
                    zero_behavior,        # 使用零行为特征
                    batch_skill_weights   # 知识点权重
                )
                learning_gain = l_w * learning_gain
                
        forget_gate = self.forget_gate(
                    knowledge_state,      # 知识状态
                    curr_interaction,     # 当前交互
                    learning_gain,        # 学习增益
                    zero_behavior,        # 使用零行为特征
                    batch_skill_weights   # 知识点权重
                )
                forget_gate = f_w * forget_gate
                
                knowledge_state = learning_gain + forget_gate * knowledge_state
            else:
                knowledge_state = self.knowledge_update(
                    prev_knowledge=knowledge_state,
                    curr_interaction=curr_interaction,
                    prev_interaction=knowledge_state,
                    interval=curr_interval,
                    behavior=zero_behavior,  # 使用零行为特征
                    skill_ids=curr_skill.long()
                )
            
            # 获取当前批次的知识点权重
            skill_weights = self.skill_weights[curr_skill.long()]
            
            # 预测下一步
            pred_input = torch.cat([knowledge_state, curr_exercise], dim=-1)
            raw_pred = self.predictor(pred_input).squeeze(-1)
            
            # 应用权重和归一化
            if learn_weights is not None:
                student_weight = learn_weights.view(-1)
            else:
                student_weight = torch.ones_like(raw_pred)
            
            compensated_pred = torch.sigmoid(
                raw_pred - torch.log(student_weight + 1e-6)
            ) * student_weight
            
            weighted_pred = compensated_pred * skill_weights
            normalized_pred = torch.clamp(
                weighted_pred / (student_weight * skill_weights + 1e-6),
                min=0.05,
                max=0.95
            )
            
            pred_list.append(normalized_pred)
        
        pred_seq = torch.stack(pred_list, dim=1)
        return pred_seq

class EKMFKT_NoLearnGate(EKMFKT):
    """不考虑学习门的变体"""
    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        """重写前向传播方法，忽略学习门"""
        # 确保输入的类型正确
        exercise_seq = exercise_seq.long()
        skill_seq = skill_seq.long()
        time_seq = time_seq.float()
        
        # 获取嵌入
        exercise_embeds = self.exercise_embed(exercise_seq)
        skill_embeds = self.skill_embed(skill_seq)
        
        # 初始化知识状态
        batch_size = exercise_seq.size(0)
        knowledge_state = torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device)
        pred_list = []
        
        for t in range(exercise_seq.size(1)):
            curr_exercise = exercise_embeds[:, t]
            curr_skill = skill_seq[:, t]
            curr_time = time_seq[:, t]
            curr_interval = interval_seq[:, t]
            
            # 计算当前时刻的交互嵌入
            curr_interaction = self.compute_interaction_embedding(
                curr_exercise,
                skill_embeds[:, t],
                response_seq[:, t],
                curr_time
            )
            
            # 计算行为特征
            curr_behavior = self.behavior_transform(
                torch.stack([attempt_seq[:, t], hint_seq[:, t]], dim=-1)
            )
            
            # 更新知识状态时只使用遗忘门
            if learn_weights is not None and forget_weights is not None:
                f_w = forget_weights.view(-1, 1)
                
                batch_skill_weights = self.skill_weights[curr_skill.long()]
                
                # 只使用遗忘门
                # 创建一个全零的learning_gain作为占位符
                dummy_learning_gain = torch.zeros_like(knowledge_state)
                
                forget_gate = self.forget_gate(
                    knowledge_state,      # 知识状态
                    curr_interaction,     # 当前交互
                    dummy_learning_gain,  # 使用零学习增益
                    curr_behavior,        # 行为特征
                    batch_skill_weights   # 知识点权重
                )
                forget_gate = f_w * forget_gate
                
                # 只应用遗忘门
                knowledge_state = forget_gate * knowledge_state
            else:
            knowledge_state = self.knowledge_update(
                    prev_knowledge=knowledge_state,
                    curr_interaction=curr_interaction,
                    prev_interaction=knowledge_state,
                    interval=curr_interval,
                    behavior=curr_behavior,
                    skill_ids=curr_skill.long()
                )
            
            # 获取当前批次的知识点权重
            skill_weights = self.skill_weights[curr_skill.long()]
            
            # 预测下一步
            pred_input = torch.cat([knowledge_state, curr_exercise], dim=-1)
            raw_pred = self.predictor(pred_input).squeeze(-1)
            
            # 应用权重和归一化
            if learn_weights is not None:
                student_weight = learn_weights.view(-1)
            else:
                student_weight = torch.ones_like(raw_pred)
            
            compensated_pred = torch.sigmoid(
                raw_pred - torch.log(student_weight + 1e-6)
            ) * student_weight
            
            weighted_pred = compensated_pred * skill_weights
            normalized_pred = torch.clamp(
                weighted_pred / (student_weight * skill_weights + 1e-6),
                min=0.05,
                max=0.95
            )
            
            pred_list.append(normalized_pred)
        
        pred_seq = torch.stack(pred_list, dim=1)
        return pred_seq

class EKMFKT_NoForgetGate(EKMFKT):
    """不考虑遗忘门的变体"""
    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        """重写前向传播方法，忽略遗忘门"""
        # 确保输入的类型正确
        exercise_seq = exercise_seq.long()
        skill_seq = skill_seq.long()
        time_seq = time_seq.float()
        
        # 获取嵌入
        exercise_embeds = self.exercise_embed(exercise_seq)
        skill_embeds = self.skill_embed(skill_seq)
        
        # 初始化知识状态
        batch_size = exercise_seq.size(0)
        knowledge_state = torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device)
        pred_list = []
        
        for t in range(exercise_seq.size(1)):
            curr_exercise = exercise_embeds[:, t]
            curr_skill = skill_seq[:, t]
            curr_time = time_seq[:, t]
            curr_interval = interval_seq[:, t]
            
            # 计算当前时刻的交互嵌入
            curr_interaction = self.compute_interaction_embedding(
                curr_exercise,
                skill_embeds[:, t],
                response_seq[:, t],
                curr_time
            )
            
            # 计算行为特征
            curr_behavior = self.behavior_transform(
                torch.stack([attempt_seq[:, t], hint_seq[:, t]], dim=-1)
            )
            
            # 更新知识状态时只使用学习门
            if learn_weights is not None and forget_weights is not None:
                l_w = learn_weights.view(-1, 1)
                
                batch_skill_weights = self.skill_weights[curr_skill.long()]
                
                # 只使用学习门
                learning_gain = self.learning_gate(
                    knowledge_state,      # 知识状态
                    curr_interval,        # 时间间隔
                    curr_interaction,     # 当前交互
                    knowledge_state,      # 占位参数，实际未使用
                    curr_behavior,        # 行为特征
                    batch_skill_weights   # 知识点权重
                )
                learning_gain = l_w * learning_gain
                
                # 只应用学习增益
                knowledge_state = learning_gain
            else:
            knowledge_state = self.knowledge_update(
                    prev_knowledge=knowledge_state,
                    curr_interaction=curr_interaction,
                    prev_interaction=knowledge_state,
                    interval=curr_interval,
                    behavior=curr_behavior,
                    skill_ids=curr_skill.long()
                )
            
            # 获取当前批次的知识点权重
            skill_weights = self.skill_weights[curr_skill.long()]
            
            # 预测下一步
            pred_input = torch.cat([knowledge_state, curr_exercise], dim=-1)
            raw_pred = self.predictor(pred_input).squeeze(-1)
            
            # 应用权重和归一化
            if learn_weights is not None:
                student_weight = learn_weights.view(-1)
            else:
                student_weight = torch.ones_like(raw_pred)
            
            compensated_pred = torch.sigmoid(
                raw_pred - torch.log(student_weight + 1e-6)
            ) * student_weight
            
            weighted_pred = compensated_pred * skill_weights
            normalized_pred = torch.clamp(
                weighted_pred / (student_weight * skill_weights + 1e-6),
                min=0.05,
                max=0.95
            )
            
            pred_list.append(normalized_pred)
        
        pred_seq = torch.stack(pred_list, dim=1)
        return pred_seq

def train_model_variant(model_class, model_name, sequences, q_matrix, device, **kwargs):
    """修改后的训练函数，加入学生分类和权重调整"""
    print(f"\nTraining {model_name}...")
    
    # 计算学生表现并分类
    student_performance = calculate_student_performance(sequences)
    student_categories = classify_students(student_performance)
    
    # 打印分类统计信息
    category_stats = {cat: sum(1 for c in student_categories.values() if c == cat) 
                     for cat in ['good', 'medium', 'poor']}
    print("\nStudent Classification Statistics:")
    for cat, count in category_stats.items():
        print(f"{cat.capitalize()}: {count} students")
    
    # 打印不同类别对应的权重设置
    print("\n当前权重设置:")
    for category in ['good', 'medium', 'poor']:
        learn_w, forget_w = get_gate_weights(category)
        print(f"类别: {category}, 学习权重: {learn_w:.3f}, 遗忘权重: {forget_w:.3f}")
    
    # 设置批次大小
    batch_size = 32
    
    # 数据划分
    indices = np.random.permutation(len(sequences))
    train_size = int(0.8 * len(sequences))
    val_size = int(0.1 * len(sequences))
    
    train_sequences = [sequences[i] for i in indices[:train_size]]
    val_sequences = [sequences[i] for i in indices[train_size:train_size+val_size]]
    test_sequences = [sequences[i] for i in indices[train_size+val_size:]]
    
    # 初始化模型
    model = model_class(**kwargs).to(device)
    
    # 初始化优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    # 初始化最佳模型记录
    best_val_metrics = {'loss': float('inf')}
    best_model = None
    
    # 记录每个类别的权重分布情况
    weight_stats = {'epoch': [], 'good_count': [], 'medium_count': [], 'poor_count': []}
    
    print("\nStarting main training...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        processed_samples = 0
        
        # 每个epoch统计各类别权重使用次数
        epoch_weight_stats = {'good': 0, 'medium': 0, 'poor': 0}
        
        # 随机选择一个批次进行权重使用详细打印
        debug_batch_idx = np.random.randint(0, len(train_sequences) // batch_size)
        
        for start_idx in range(0, len(train_sequences), batch_size):
            end_idx = min(start_idx + batch_size, len(train_sequences))
            current_batch = train_sequences[start_idx:end_idx]
            current_batch_size = len(current_batch)
            
            # 获取当前批次中每个学生的门控权重
            learn_weights = []
            forget_weights = []
            batch_categories = []
            
            for seq in current_batch:
                student_id = seq['student_id']
                category = student_categories.get(student_id, 'medium')
                batch_categories.append(category)
                epoch_weight_stats[category] += 1
                
                l_w, f_w = get_gate_weights(category)
                learn_weights.append(l_w)
                forget_weights.append(f_w)
            
            # 转换为张量并移到正确的设备上
            learn_weights = torch.tensor(learn_weights, device=device).float()
            forget_weights = torch.tensor(forget_weights, device=device).float()
            
            # 在随机选择的批次上打印详细权重信息
            batch_num = start_idx // batch_size
            if batch_num == debug_batch_idx and epoch % 10 == 0:
                print(f"\nEpoch {epoch+1}，批次 {batch_num + 1} 的详细权重信息:")
                print("学生ID\t类别\t\t学习权重\t遗忘权重")
                print("-" * 50)
                for i, seq in enumerate(current_batch):
                    student_id = seq['student_id']
                    print(f"{student_id}\t{batch_categories[i]}\t\t{learn_weights[i].item():.3f}\t\t{forget_weights[i].item():.3f}")
            
            batch = prepare_batch(current_batch, current_batch_size, device)
            
            # 将权重传递给模型
            pred_seq = model(
                exercise_seq=batch['exercise_seq'],
                skill_seq=batch['skill_seq'],
                response_seq=batch['response_seq'],
                time_seq=batch['time_seq'],
                interval_seq=batch['interval_seq'],
                attempt_seq=batch['attempt_seq'],
                hint_seq=batch['hint_seq'],
                q_matrix=q_matrix,
                learn_weights=learn_weights,
                forget_weights=forget_weights
            )
            
            loss = model.loss(pred_seq, batch['response_seq'], batch['mask_seq'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            processed_samples += current_batch_size
        
        # 记录每个epoch的类别统计
        weight_stats['epoch'].append(epoch + 1)
        weight_stats['good_count'].append(epoch_weight_stats['good'])
        weight_stats['medium_count'].append(epoch_weight_stats['medium'])
        weight_stats['poor_count'].append(epoch_weight_stats['poor'])
        
        print(f"Processed {processed_samples} samples in epoch {epoch+1}")
        print(f"权重使用统计 - 好: {epoch_weight_stats['good']}, 中: {epoch_weight_stats['medium']}, 差: {epoch_weight_stats['poor']}")
        
        train_loss = total_loss / len(train_sequences)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_model(model, val_sequences, q_matrix, device)
        
        if val_metrics['loss'] < best_val_metrics['loss']:
            best_val_metrics = val_metrics
            best_model = copy.deepcopy(model)
        
        print(f"Epoch [{epoch+1}/50] - Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
              f"Val AUC: {val_metrics['AUC']:.4f}, Val ACC: {val_metrics['ACC']:.4f}, Val RMSE: {val_metrics['RMSE']:.4f}")
        
        scheduler.step()
        
        # 在每个epoch结束后更新学生类别
        if epoch > 0 and epoch % 5 == 0:
            print("\n更新学生分类...")
            new_student_performance = calculate_student_performance(train_sequences)
            student_categories = classify_students(new_student_performance)
            
            new_category_stats = {cat: sum(1 for c in student_categories.values() if c == cat) 
                                for cat in ['good', 'medium', 'poor']}
            print("\nUpdated Student Classification at Epoch", epoch)
            for cat, count in new_category_stats.items():
                print(f"{cat.capitalize()}: {count} students")
    
    # 打印权重使用统计图表
    print("\n训练过程中权重使用统计:")
    print("Epoch\t好\t中\t差")
    for i in range(len(weight_stats['epoch'])):
        print(f"{weight_stats['epoch'][i]}\t{weight_stats['good_count'][i]}\t{weight_stats['medium_count'][i]}\t{weight_stats['poor_count'][i]}")
    
    # 在测试集上评估最佳模型
    test_metrics = evaluate_model(best_model, test_sequences, q_matrix, device)
    
    return best_model, test_metrics

def evaluate_model(model, sequences, q_matrix, device, batch_size=32):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    num_valid_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = prepare_batch(sequences[i:i+batch_size], batch_size, device)
            mask = batch['mask_seq'].bool()
            
            if mask.sum() == 0:
                continue
                
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
            
            pred_seq = torch.sigmoid(pred_seq)
            pred_seq = torch.clamp(pred_seq, 1e-7, 1-1e-7)
            
            valid_preds = pred_seq[mask]
            valid_targets = batch['response_seq'][mask]
            
            all_preds.extend(valid_preds.cpu().numpy())
            all_targets.extend(valid_targets.cpu().numpy())
            
            loss = model.loss(pred_seq, batch['response_seq'], batch['mask_seq'])
            if torch.isfinite(loss):
                total_loss += loss.item()
                num_valid_batches += 1
    
    if not all_preds or not all_targets:
        return {'loss': float('inf'), 'AUC': 0.0, 'ACC': 0.0, 'RMSE': float('inf')}
    
    metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
    
    if num_valid_batches > 0:
        metrics['loss'] = total_loss / num_valid_batches
    else:
        metrics['loss'] = float('inf')
    
    return metrics

def calculate_metrics(y_true, y_pred):
    """计算模型性能指标
    
    计算三个主要指标：
    - AUC: ROC曲线下面积
    - ACC: 准确率（使用最优阈值）
    - RMSE: 均方根误差
    """
    # 将张量转换为numpy数组
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # 计算AUC
    auc = roc_auc_score(y_true, y_pred)
    
    # 寻找最优阈值
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    # 使用最优阈值进行分类
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'AUC': auc,
        'ACC': acc,
        'RMSE': rmse,
        'threshold': optimal_threshold
    }

def main():
    # 设置随机种子确保数据一致性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    sequences, q_matrix, num_exercises, num_skills = process_data(
        'student_log_11.csv',
        max_seq_len=100
    )
    q_matrix = torch.tensor(q_matrix, dtype=torch.float)
    
    # 强制使用CPU
    device = torch.device('cpu')
    q_matrix = q_matrix.to(device)
    
    # 模型参数
    model_params = {
        'num_exercises': num_exercises,
        'num_skills': num_skills,
        'hidden_dim': 128,
        'embed_dim': 128
    }
    
    # 训练所有变体
    variants = [
        (EKMFKT, 'weighted_model'),
        (EKMFKT_NoResponseTime, 'no_response_time'),
        (EKMFKT_NoIntervalTime, 'no_interval_time'),
        (EKMFKT_NoBehavior, 'no_behavior'),
        (EKMFKT_NoForgetGate, 'no_forget_gate'),
        (EKMFKT_NoLearnGate, 'no_learn_gate')
    ]
    
    results = {}
    for model_class, model_name in variants:
        model, test_metrics = train_model_variant(
            model_class, 
            model_name, 
            sequences,
            q_matrix, 
            device, 
            **model_params
        )
        results[model_name] = test_metrics
        
        # 保存每个变体的模型
        torch.save(model.state_dict(), f'{model_name}.pth')
    
    # 打印更详细的比较结果
    print("\nAblation Study Results:")
    print("\nModel\t\tAUC\t\tACC\t\tRMSE")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['AUC']:.4f}\t{metrics['ACC']:.4f}\t{metrics['RMSE']:.4f}")

if __name__ == "__main__":
    main()

