import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import networkx as nx
import random

class PretrainEmbedding(nn.Module):
    """
    试题-知识点异构图嵌入预训练类
    
    该类用于:
    1. 构建试题-知识点的异构图
    2. 基于随机游走生成序列
    3. 预训练试题和知识点的嵌入表示
    """
    def __init__(self,
                 num_exercises: int,    # 试题总数
                 num_skills: int,       # 知识点总数
                 embed_dim: int,        # 嵌入维度
                 q_matrix: np.ndarray): # Q矩阵(试题-知识点关系矩阵)
        """
        初始化预训练模块
        
        参数说明:
        - num_exercises: 试题的总数量
        - num_skills: 知识点的总数量
        - embed_dim: 嵌入向量的维度
        - q_matrix: 形状为[num_exercises, num_skills]的二维数组，表示试题和知识点的关联关系
        """
        super().__init__()
        self.num_exercises = num_exercises
        self.num_skills = num_skills
        self.embed_dim = embed_dim
        # 确保q_matrix是float类型的tensor
        self.q_matrix = q_matrix.float()
        
        # 构建异构图
        self.graph = nx.Graph()
        self._build_graph()
        
        # 只保留S-E-S元路径
        self.meta_paths = [
            ['skill', 'exercise', 'skill']  # S-E-S路径：知识点->试题->知识点
        ]
        
        # 初始化嵌入层
        self.exercise_embed = nn.Embedding(num_exercises, embed_dim)
        self.skill_embed = nn.Embedding(num_skills, embed_dim)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.exercise_embed.weight)
        nn.init.xavier_uniform_(self.skill_embed.weight)
        
    def _build_graph(self):
        """
        构建试题-知识点异构图
        
        图的结构:
        - 节点: 试题节点和知识点节点
        - 边: 基于Q矩阵中的关联关系
        """
        # 添加试题节点
        for i in range(self.num_exercises):
            self.graph.add_node(i, type='exercise')
            
        # 添加知识点节点
        for i in range(self.num_skills):
            self.graph.add_node(self.num_exercises + i, type='skill')
            
        # 根据Q矩阵添加边
        for i in range(self.num_exercises):
            for j in range(self.num_skills):
                if self.q_matrix[i, j] == 1:
                    self.graph.add_edge(i, self.num_exercises + j)
                    
    def _meta_path_sampling(self, start_node, meta_path):
        """基于元路径的采样"""
        path = [start_node]
        current = start_node
        
        for node_type in meta_path[1:]:
            neighbors = [n for n in self.graph.neighbors(current) 
                       if self.graph.nodes[n]['type'] == node_type]
            if not neighbors:
                return None
            current = random.choice(neighbors)
            path.append(current)
        
        return path
    
    def generate_pairs(self):
        """生成训练对"""
        exercise_skills = []
        q_matrix_np = self.q_matrix.cpu().numpy()
        
        for i in range(self.num_exercises):
            skills = np.where(q_matrix_np[i] > 0)[0]
            if len(skills) > 0:
                for skill in skills:
                    if skill < self.num_skills:
                        exercise_skills.append([int(i), int(skill)])
        
        if not exercise_skills:
            raise ValueError("No valid exercise-skill pairs found!")
        
        return np.array(exercise_skills, dtype=np.int64)
    
    def to(self, device):
        """
        将模型移动到指定设备(CPU/GPU)
        """
        self.exercise_embed = self.exercise_embed.to(device)
        self.skill_embed = self.skill_embed.to(device)
        return self
        
    def forward(self, pairs):
        """前向传播"""
        # 确保pairs是long类型的tensor
        if not isinstance(pairs, torch.Tensor):
            pairs = torch.tensor(pairs, dtype=torch.long, device=self.exercise_embed.weight.device)
        else:
            pairs = pairs.to(dtype=torch.long, device=self.exercise_embed.weight.device)
        
        # 分离并确保是long类型，同时确保索引在有效范围内
        exercise_ids = torch.clamp(pairs[:, 0].long(), 0, self.num_exercises - 1)
        skill_ids = torch.clamp(pairs[:, 1].long(), 0, self.num_skills - 1)
        
        try:
            # 获取嵌入
            exercise_embeds = self.exercise_embed(exercise_ids)
            skill_embeds = self.skill_embed(skill_ids)
            
            # 计算相似度
            similarity = torch.sum(exercise_embeds * skill_embeds, dim=1)
            return torch.sigmoid(similarity)
        except Exception as e:
            print(f"Error in forward pass:")
            print(f"exercise_ids shape: {exercise_ids.shape}, dtype: {exercise_ids.dtype}")
            print(f"skill_ids shape: {skill_ids.shape}, dtype: {skill_ids.dtype}")
            print(f"exercise_ids range: [{exercise_ids.min()}, {exercise_ids.max()}]")
            print(f"skill_ids range: [{skill_ids.min()}, {skill_ids.max()}]")
            raise e
    
    def train_step(self, optimizer, pairs):
        """训练一步"""
        # 确保pairs是long类型且在正确的设备上
        if not isinstance(pairs, torch.Tensor):
            pairs = torch.tensor(pairs, dtype=torch.long, device=self.exercise_embed.weight.device)
        else:
            pairs = pairs.to(dtype=torch.long, device=self.exercise_embed.weight.device)
        
        # 生成负样本
        num_pairs = len(pairs)
        negative_pairs = []
        q_matrix_np = self.q_matrix.cpu().numpy()
        
        for i in range(num_pairs):
            exercise_id = int(pairs[i, 0].item())  # 确保是Python整数
            exercise_id = min(exercise_id, self.num_exercises - 1)  # 确保不超出范围
            
            # 随机选择一个不相关的知识点作为负样本
            available_skills = np.where(q_matrix_np[exercise_id] == 0)[0]
            if len(available_skills) > 0:
                neg_skill = int(np.random.choice(available_skills))
            else:
                neg_skill = int(np.random.randint(0, self.num_skills - 1))
            
            negative_pairs.append([exercise_id, neg_skill])
        
        # 转换为tensor并确保是long类型
        negative_pairs = torch.tensor(negative_pairs, dtype=torch.long, device=pairs.device)
        
        try:
            # 计算正样本和负样本的相似度
            pos_similarity = self.forward(pairs)
            neg_similarity = self.forward(negative_pairs)
            
            # 计算对比损失
            loss = -torch.mean(torch.log(pos_similarity + 1e-8) + torch.log(1 - neg_similarity + 1e-8))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss.item()
        except Exception as e:
            print(f"Error in train_step:")
            print(f"pairs shape: {pairs.shape}, dtype: {pairs.dtype}")
            print(f"negative_pairs shape: {negative_pairs.shape}, dtype: {negative_pairs.dtype}")
            print(f"pairs range: [{pairs.min()}, {pairs.max()}]")
            print(f"negative_pairs range: [{negative_pairs.min()}, {negative_pairs.max()}]")
            raise e
    
    def get_embeddings(self):
        """获取预训练的嵌入"""
        return self.exercise_embed.weight.detach(), self.skill_embed.weight.detach() 