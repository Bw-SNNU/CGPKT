import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from main import EKMFKT

def test_model():
    """测试模型是否能正常初始化和前向传播"""
    print("开始测试模型...")
    
    # 创建更小的模型用于调试
    num_exercises = 10  # 降低维度
    num_skills = 5      # 降低维度
    hidden_dim = 8      # 降低维度
    embed_dim = 8       # 降低维度
    batch_size = 2      # 减小批次大小
    seq_len = 3         # 减小序列长度
    
    # 初始化模型
    print("初始化模型...")
    model = EKMFKT(
        num_exercises=num_exercises,
        num_skills=num_skills,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        dropout=0.1
    )
    
    print("\n模型参数:")
    print(f"- 试题总数: {num_exercises}")
    print(f"- 知识点总数: {num_skills}")
    print(f"- 隐藏层维度: {hidden_dim}")
    print(f"- 嵌入维度: {embed_dim}")
    
    print("\n模型结构:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"- {name}: {param.shape}")
    
    # 创建随机输入
    print("\n创建随机输入...")
    exercise_seq = torch.randint(0, num_exercises, (batch_size, seq_len))
    skill_seq = torch.randint(0, num_skills, (batch_size, seq_len))
    response_seq = torch.randint(0, 2, (batch_size, seq_len)).float()
    time_seq = torch.rand(batch_size, seq_len)
    interval_seq = torch.rand(batch_size, seq_len)
    attempt_seq = torch.rand(batch_size, seq_len)
    hint_seq = torch.rand(batch_size, seq_len)
    q_matrix = torch.rand(num_exercises, num_skills)
    mask_seq = torch.ones(batch_size, seq_len)
    
    print("\n张量形状:")
    print(f"- exercise_seq: {exercise_seq.shape}")
    print(f"- skill_seq: {skill_seq.shape}")
    print(f"- response_seq: {response_seq.shape}")
    print(f"- time_seq: {time_seq.shape}")
    print(f"- interval_seq: {interval_seq.shape}")
    print(f"- attempt_seq: {attempt_seq.shape}")
    print(f"- hint_seq: {hint_seq.shape}")
    print(f"- q_matrix: {q_matrix.shape}")
    
    # 测试学习和遗忘权重
    learn_weights = torch.ones(batch_size) * 1.15
    forget_weights = torch.ones(batch_size) * 0.85
    
    try:
        # 测试各个组件
        print("\n逐步测试各个组件...")
        
        # 1. 测试嵌入层
        print("1. 测试嵌入层...")
        exercise_embeds = model.exercise_embed(exercise_seq)
        skill_embeds = model.skill_embed(skill_seq)
        print(f"- exercise_embeds shape: {exercise_embeds.shape}")
        print(f"- skill_embeds shape: {skill_embeds.shape}")
        
        # 2. 测试交互嵌入计算
        print("2. 测试交互嵌入计算...")
        curr_exercise = exercise_embeds[:, 0]
        curr_skill = skill_embeds[:, 0]
        curr_response = response_seq[:, 0]
        curr_time = time_seq[:, 0]
        
        curr_interaction = model.compute_interaction_embedding(
            curr_exercise, 
            curr_skill, 
            curr_response, 
            curr_time
        )
        print(f"- curr_interaction shape: {curr_interaction.shape}")
        
        # 3. 测试行为特征转换
        print("3. 测试行为特征转换...")
        curr_behavior = model.behavior_transform(
            torch.stack([attempt_seq[:, 0], hint_seq[:, 0]], dim=-1)
        )
        print(f"- curr_behavior shape: {curr_behavior.shape}")
        
        # 4. 测试学习门和遗忘门
        print("4. 测试学习门和遗忘门...")
        batch_skill_weights = model.skill_weights[skill_seq[:, 0]]
        knowledge_state = torch.zeros(batch_size, hidden_dim)
        
        learning_gain = model.learning_gate(
            knowledge_state,
            interval_seq[:, 0],
            curr_interaction,
            knowledge_state,
            curr_behavior,
            batch_skill_weights
        )
        print(f"- learning_gain shape: {learning_gain.shape}")
        
        forget_gate = model.forget_gate(
            knowledge_state,
            curr_interaction,
            learning_gain,
            curr_behavior,
            batch_skill_weights
        )
        print(f"- forget_gate shape: {forget_gate.shape}")
        
        # 5. 测试预测
        print("5. 测试预测...")
        pred_input = torch.cat([knowledge_state, curr_exercise], dim=-1)
        print(f"- pred_input shape: {pred_input.shape}")
        
        raw_pred = model.predictor(pred_input)
        print(f"- raw_pred shape: {raw_pred.shape}")
        
        # 前向传播
        print("\n测试完整前向传播...")
        outputs = model(
            exercise_seq=exercise_seq,
            skill_seq=skill_seq,
            response_seq=response_seq,
            time_seq=time_seq,
            interval_seq=interval_seq,
            attempt_seq=attempt_seq,
            hint_seq=hint_seq,
            q_matrix=q_matrix,
            learn_weights=learn_weights,
            forget_weights=forget_weights
        )
        
        # 测试损失函数
        print("\n测试损失函数...")
        loss = model.loss(outputs, response_seq, mask_seq)
        
        print(f"输出形状: {outputs.shape}")
        print(f"损失值: {loss.item()}")
        print("\n测试成功！模型可以正常运行。")
        
    except Exception as e:
        print(f"\n测试失败，错误信息: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model() 