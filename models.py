import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import calculate_skill_weights

class WeightedLearningGate(nn.Module):
    """带权重的学习门模块"""
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        total_input_dim = hidden_dim + 1 + embed_dim + hidden_dim + hidden_dim
        self.linear = nn.Linear(total_input_dim, hidden_dim)
        self.b5 = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, prev_interaction, interval, curr_interaction, prev_knowledge, behavior, skill_weights):
        combined_input = torch.cat([
            prev_interaction,
            interval.unsqueeze(-1),
            curr_interaction,
            prev_knowledge,
            behavior
        ], dim=-1)
        
        gate = torch.sigmoid(self.linear(combined_input) + self.b5)
        skill_weights = skill_weights.view(-1, 1).expand(-1, gate.size(1))
        weighted_gate = gate * skill_weights
        return weighted_gate

class WeightedForgetGate(nn.Module):
    """带权重的遗忘门"""
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        total_input_dim = hidden_dim + embed_dim + hidden_dim + hidden_dim
        self.linear = nn.Linear(total_input_dim, hidden_dim)
        self.b6 = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, prev_interaction, curr_interaction, learning_gain, behavior, skill_weights):
        combined_input = torch.cat([
            prev_interaction,
            curr_interaction,
            learning_gain,
            behavior
        ], dim=-1)
        
        gate = torch.sigmoid(self.linear(combined_input) + self.b6)
        skill_weights = skill_weights.view(-1, 1).expand(-1, gate.size(1))
        weighted_gate = gate * (1 - skill_weights)
        return weighted_gate

class EKMFKT(nn.Module):
    """基础知识追踪模型"""
    def __init__(self, num_exercises, num_skills, hidden_dim=128, embed_dim=128, dropout=0.2):
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
        self.exercise_embed = nn.Embedding(num_exercises, embed_dim)
        self.skill_embed = nn.Embedding(num_skills, embed_dim)
        self.response_embed = nn.Embedding(2, embed_dim)
        
        # 初始化嵌入层
        nn.init.xavier_uniform_(self.exercise_embed.weight)
        nn.init.xavier_uniform_(self.skill_embed.weight)
        nn.init.xavier_uniform_(self.response_embed.weight)
        
        # 其他层
        self.time_transform = nn.Linear(1, embed_dim)
        self.interaction_transform = nn.Linear(embed_dim * 4, embed_dim)
        self.behavior_transform = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 门控模块
        self.learning_gate = nn.Linear(hidden_dim + embed_dim + 1 + hidden_dim + hidden_dim, hidden_dim)
        self.forget_gate = nn.Linear(hidden_dim + embed_dim + hidden_dim + hidden_dim, hidden_dim)
        
        # 知识状态转换
        self.knowledge_transform = nn.Linear(hidden_dim + embed_dim + 1 + hidden_dim + hidden_dim, hidden_dim)
        
        # 预测模块
        self.predictor = nn.Linear(hidden_dim + embed_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        batch_size, seq_len = exercise_seq.size()
        
        # 获取嵌入
        exercise_embed = self.exercise_embed(exercise_seq)
        skill_embed = self.skill_embed(skill_seq)
        response_embed = self.response_embed(response_seq.long())
        
        # 转换时间特征
        time_embed = self.time_transform(time_seq.unsqueeze(-1))
        
        # 计算交互嵌入
        interaction = torch.cat([
            exercise_embed,
            skill_embed,
            time_embed,
            response_embed
        ], dim=-1)
        interaction = F.relu(self.interaction_transform(interaction))
        
        # 计算行为特征
        behavior = torch.stack([attempt_seq, hint_seq], dim=-1)
        behavior = self.behavior_transform(behavior)
        
        # 初始化知识状态
        knowledge_state = torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device)
        
        # 存储预测结果
        predictions = []
        
        # 序列处理
        for t in range(seq_len):
            # 当前时刻的特征
            curr_exercise = exercise_embed[:, t]
            curr_skill = skill_embed[:, t]
            curr_interaction = interaction[:, t]
            curr_behavior = behavior[:, t]
            curr_interval = interval_seq[:, t]
            
            # 学习门
            learn_input = torch.cat([
                knowledge_state,
                curr_interaction,
                curr_interval.unsqueeze(-1),
                curr_behavior,
                behavior[:, t] if t < seq_len - 1 else torch.zeros_like(curr_behavior)
            ], dim=-1)
            learn_gate = torch.sigmoid(self.learning_gate(learn_input))
            
            if learn_weights is not None:
                learn_gate = learn_gate * learn_weights.unsqueeze(-1)
            
            # 遗忘门
            forget_input = torch.cat([
                knowledge_state,
                curr_interaction,
                learn_gate,
                curr_behavior
            ], dim=-1)
            forget_gate = torch.sigmoid(self.forget_gate(forget_input))
            
            if forget_weights is not None:
                forget_gate = forget_gate * forget_weights.unsqueeze(-1)
            
            # 更新知识状态
            knowledge_state = learn_gate + forget_gate * knowledge_state
            
            # 预测
            pred_input = torch.cat([knowledge_state, curr_exercise], dim=-1)
            pred = torch.sigmoid(self.predictor(pred_input))
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1).squeeze(-1)

    def loss(self, pred_seq, target_seq, mask_seq):
        pred_seq = torch.clamp(pred_seq, 1e-7, 1-1e-7)
        loss = F.binary_cross_entropy(
            pred_seq,
            target_seq,
            reduction='none'
        )
        loss = loss * mask_seq
        return loss.sum() / (mask_seq.sum() + 1e-8)

class EKMFKT_NoResponseTime(EKMFKT):
    """不使用响应时间的变体"""
    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        # 将time_seq设为0，相当于不使用响应时间信息
        time_seq = torch.zeros_like(time_seq)
        return super().forward(exercise_seq, skill_seq, response_seq, time_seq, interval_seq,
                             attempt_seq, hint_seq, q_matrix, learn_weights, forget_weights)

class EKMFKT_NoIntervalTime(EKMFKT):
    """不使用间隔时间的变体"""
    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        # 将interval_seq设为0，相当于不使用间隔时间信息
        interval_seq = torch.zeros_like(interval_seq)
        return super().forward(exercise_seq, skill_seq, response_seq, time_seq, interval_seq,
                             attempt_seq, hint_seq, q_matrix, learn_weights, forget_weights)

class EKMFKT_NoBehavior(EKMFKT):
    """不使用行为特征的变体"""
    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        # 将attempt_seq和hint_seq设为0，相当于不使用行为特征
        attempt_seq = torch.zeros_like(attempt_seq)
        hint_seq = torch.zeros_like(hint_seq)
        return super().forward(exercise_seq, skill_seq, response_seq, time_seq, interval_seq,
                             attempt_seq, hint_seq, q_matrix, learn_weights, forget_weights)

class EKMFKT_NoForgetGate(EKMFKT):
    """不使用遗忘门的变体"""
    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        batch_size, seq_len = exercise_seq.size()
        
        # 基本特征处理与父类相同
        exercise_embed = self.exercise_embed(exercise_seq)
        skill_embed = self.skill_embed(skill_seq)
        response_embed = self.response_embed(response_seq.long())
        time_embed = self.time_transform(time_seq.unsqueeze(-1))
        
        interaction = torch.cat([
            exercise_embed,
            skill_embed,
            time_embed,
            response_embed
        ], dim=-1)
        interaction = F.relu(self.interaction_transform(interaction))
        
        behavior = torch.stack([attempt_seq, hint_seq], dim=-1)
        behavior = self.behavior_transform(behavior)
        
        knowledge_state = torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device)
        predictions = []
        
        for t in range(seq_len):
            curr_exercise = exercise_embed[:, t]
            curr_interaction = interaction[:, t]
            curr_behavior = behavior[:, t]
            curr_interval = interval_seq[:, t]
            
            # 只使用学习门
            learn_input = torch.cat([
                knowledge_state,
                curr_interaction,
                curr_interval.unsqueeze(-1),
                curr_behavior,
                behavior[:, t] if t < seq_len - 1 else torch.zeros_like(curr_behavior)
            ], dim=-1)
            learn_gate = torch.sigmoid(self.learning_gate(learn_input))
            
            if learn_weights is not None:
                learn_gate = learn_gate * learn_weights.unsqueeze(-1)
            
            # 直接更新知识状态，不使用遗忘门
            knowledge_state = learn_gate
            
            pred_input = torch.cat([knowledge_state, curr_exercise], dim=-1)
            pred = torch.sigmoid(self.predictor(pred_input))
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1).squeeze(-1)

class EKMFKT_NoLearnGate(EKMFKT):
    """不使用学习门的变体"""
    def forward(self, exercise_seq, skill_seq, response_seq, time_seq, interval_seq, 
                attempt_seq, hint_seq, q_matrix, learn_weights=None, forget_weights=None):
        batch_size, seq_len = exercise_seq.size()
        
        # 基本特征处理与父类相同
        exercise_embed = self.exercise_embed(exercise_seq)
        skill_embed = self.skill_embed(skill_seq)
        response_embed = self.response_embed(response_seq.long())
        time_embed = self.time_transform(time_seq.unsqueeze(-1))
        
        interaction = torch.cat([
            exercise_embed,
            skill_embed,
            time_embed,
            response_embed
        ], dim=-1)
        interaction = F.relu(self.interaction_transform(interaction))
        
        behavior = torch.stack([attempt_seq, hint_seq], dim=-1)
        behavior = self.behavior_transform(behavior)
        
        knowledge_state = torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device)
        predictions = []
        
        for t in range(seq_len):
            curr_exercise = exercise_embed[:, t]
            curr_interaction = interaction[:, t]
            curr_behavior = behavior[:, t]
            
            # 只使用遗忘门
            forget_input = torch.cat([
                knowledge_state,
                curr_interaction,
                torch.zeros(batch_size, self.hidden_dim, device=exercise_seq.device),  # 替代学习门的输出
                curr_behavior
            ], dim=-1)
            forget_gate = torch.sigmoid(self.forget_gate(forget_input))
            
            if forget_weights is not None:
                forget_gate = forget_gate * forget_weights.unsqueeze(-1)
            
            # 直接更新知识状态，不使用学习门
            knowledge_state = forget_gate * knowledge_state
            
            pred_input = torch.cat([knowledge_state, curr_exercise], dim=-1)
            pred = torch.sigmoid(self.predictor(pred_input))
            predictions.append(pred)
        
        return torch.stack(predictions, dim=1).squeeze(-1)

# 添加其他模型变体类（EKMFKT_NoResponseTime, EKMFKT_NoIntervalTime等）
# ... (保持原有的模型变体类实现不变) 