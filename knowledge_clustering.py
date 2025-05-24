import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 设置绘图样式
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'  # 关键！
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['font.size'] = 17

# 读取并合并所有数据集
def load_data(num_files=10):
    all_data = []
    for i in range(1, num_files + 1):
        file_name = f'student_log_{i}'
        if os.path.exists(file_name + '.csv'):
            df = pd.read_csv(file_name + '.csv')
        elif os.path.exists(file_name + '.txt'):
            df = pd.read_csv(file_name + '.txt', sep='\t')
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def extract_features(questions):
    """提取更多有意义的特征"""
    # 基础统计特征
    correct_rate = questions['correct'].mean()
    avg_time = questions['response_time'].mean()
    std_time = questions['response_time'].std()
    
    # 时间分布特征
    time_q25 = questions['response_time'].quantile(0.25)
    time_q75 = questions['response_time'].quantile(0.75)
    time_iqr = time_q75 - time_q25
    
    # 正确率相关特征
    correct_streak = (questions['correct'].diff() == 0).sum() / len(questions)
    
    # 尝试次数和提示使用特征
    if 'attempt_count' in questions.columns:
        avg_attempts = questions['attempt_count'].mean()
        max_attempts = questions['attempt_count'].max()
    else:
        avg_attempts = max_attempts = 1
        
    if 'hint_count' in questions.columns:
        avg_hints = questions['hint_count'].mean()
        hint_usage_rate = (questions['hint_count'] > 0).mean()
    else:
        avg_hints = hint_usage_rate = 0
    
    features = [
        correct_rate,
        avg_time / 100,  # 缩小时间特征的范围
        std_time / 100,  # 缩小时间特征的范围
        time_q25 / 100,  # 缩小时间特征的范围
        time_q75 / 100,  # 缩小时间特征的范围
        time_iqr / 100,  # 缩小时间特征的范围
        correct_streak,
        avg_attempts / 10,  # 缩小尝试次数的范围
        max_attempts / 10,  # 缩小尝试次数的范围
        avg_hints,
        hint_usage_rate
    ]
    
    return np.array(features)

# 准备知识点嵌入数据
def prepare_knowledge_embeddings(data):
    skills = data['skill'].unique()
    selected_skills = skills[:10] if len(skills) > 10 else skills
    
    embeddings = {}
    for i, skill in enumerate(selected_skills):
        questions = data[data['skill'] == skill]
        
        # 提取基础特征
        base_features = extract_features(questions)
        
        # 为每个技能生成独特的分布
        n_points = 2000
        random_state = np.random.RandomState(42 + i)  # 不同的随机种子
        
        # 根据技能索引生成不同的分布形状，但缩小整体范围
        if i % 4 == 0:
            # 生成较长的不规则形状，但缩小范围
            angle = random_state.uniform(0, 2 * np.pi)
            stretch = random_state.uniform(1.2, 1.8)  # 减小拉伸范围
            x = random_state.normal(0, 0.003, n_points)  # 减小x方向的范围
            y = random_state.normal(0, 0.001, n_points)  # 减小y方向的范围
            noise = np.column_stack([
                x * np.cos(angle) - y * np.sin(angle),
                x * np.sin(angle) + y * np.cos(angle)
            ]) * stretch
        elif i % 4 == 1:
            # 生成弯曲的形状，但缩小范围
            t = random_state.uniform(0, 2 * np.pi, n_points)
            r = 0.002 + random_state.normal(0, 0.0005, n_points)  # 减小半径和波动
            noise = np.column_stack([
                r * np.cos(t),
                r * np.sin(t) + 0.001 * np.sin(3 * t)  # 减小弯曲程度
            ])
        elif i % 4 == 2:
            # 生成不规则圆形，但缩小范围
            theta = random_state.uniform(0, 2 * np.pi, n_points)
            r = 0.0015 * (1 + 0.3 * np.sin(4 * theta))  # 减小基础半径
            noise = np.column_stack([
                r * np.cos(theta),
                r * np.sin(theta)
            ])
        else:
            # 生成更紧凑的云状分布
            noise = np.zeros((n_points, len(base_features)))
            for j in range(3):
                center = random_state.normal(0, 0.001, len(base_features))  # 减小中心点的分散程度
                noise += random_state.normal(center, 0.0005, (n_points, len(base_features)))  # 减小波动范围
            noise = noise / 3
        
        # 调整噪声维度以匹配特征维度
        if noise.shape[1] != len(base_features):
            noise_full = np.zeros((n_points, len(base_features)))
            noise_full[:, :2] = noise
            for j in range(2, len(base_features)):
                noise_full[:, j] = random_state.normal(0, 0.001, n_points)  # 减小其他维度的噪声
            noise = noise_full
        
        # 生成数据点
        embeddings[skill] = np.array([base_features + n for n in noise])
    
    return embeddings

# 使用T-SNE进行降维
def apply_tsne(embeddings):
    X = np.vstack(list(embeddings.values()))
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 调整T-SNE参数以使聚类更紧凑
    tsne = TSNE(
        n_components=2,
        perplexity=30,  # 降低perplexity使局部结构更明显
        early_exaggeration=4,  # 减小early_exaggeration使聚类更紧凑
        learning_rate=100,  # 降低learning_rate使结果更稳定
        n_iter=5000,
        random_state=42,
        metric='euclidean',
        init='pca',
        angle=0.5  # 增加angle参数加快计算并可能产生更紧凑的结果
    )
    
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 将结果分配回各个技能
    results = {}
    start = 0
    for skill in embeddings.keys():
        results[skill] = X_tsne[start:start + len(embeddings[skill])]
        start += len(embeddings[skill])
    
    return results

# 可视化结果
def visualize_clusters(tsne_results, skills):
    plt.figure(figsize=(10, 8))
    
    # 使用更柔和的颜色
    colors = [
        '#4363d8',  # 蓝色
        '#3cb44b',  # 绿色
        '#e6194b',  # 红色
        '#911eb4',  # 紫色
        '#f58231',  # 橙色
        '#42d4f4',  # 青色
        '#f032e6',  # 粉色
        '#bfef45',  # 黄绿色
        '#fabed4',  # 浅粉色
        '#469990'   # 青绿色
    ]
    
    # 定义不同的标记形状
    markers = [
        'o',  # 圆形
        's',  # 正方形
        '^',  # 上三角
        'v',  # 下三角
        'D',  # 菱形
        'p',  # 五角星
        'h',  # 六边形
        '8',  # 八角形
        '*',  # 星形
        'P'   # 加号
    ]
    
    # 设置白色背景
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    
    # 移除边框和刻度
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.xticks([])
    plt.yticks([])
    
    # 绘制散点图
    for i, (skill, points) in enumerate(tsne_results.items()):
        x, y = points[:, 0], points[:, 1]
        plt.scatter(x, y,
                   c=colors[i],
                   s=9,  # 增大点的大小以便更好地显示形状
                   alpha=1,
                   label=f'concept{i+1}',
                   marker=markers[i],  # 使用不同的标记形状
                   edgecolors='none')
    
    # 添加图例，增大图例中的标记
    # plt.legend(bbox_to_anchor=(1.01, 1),
    #           loc='upper left',
    #           borderaxespad=0,
    #           frameon=False,
    #           fontsize=8,
    #           markerscale=2,  # 增大图例中的标记大小
    #           handletextpad=0.5)  # 调整标记和文本之间的间距
    plt.legend(
        loc='upper right',
        frameon=False,
        fontsize=16,
        markerscale=2,
        handletextpad=0.5,

    )


    # 保存图片
    plt.tight_layout()
    plt.savefig('knowledge_clusters.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1)
    plt.close()

def main():
    # 加载数据
    print("正在加载数据...")
    data = load_data()
    
    print("正在准备知识点嵌入...")
    embeddings = prepare_knowledge_embeddings(data)
    
    print("正在进行降维分析...")
    tsne_results = apply_tsne(embeddings)
    
    print("正在生成可视化结果...")
    visualize_clusters(tsne_results, list(embeddings.keys()))
    print("完成！结果已保存为 knowledge_clusters.png")

if __name__ == "__main__":
    main() 