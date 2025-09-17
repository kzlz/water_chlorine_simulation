import wntr
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import random

# --- 1. 配置与超参数 ---
# 设置中文字体，用于绘图
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("警告：未找到中文字体'SimHei'，图表中的中文可能无法正常显示。")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {DEVICE}")

# 模型超参数
LOOK_BACK = 12  # 用过去12个时间步（3小时）的数据
HIDDEN_DIM_GNN = 32  # GNN 的隐藏层维度
HIDDEN_DIM_GRU = 64  # GRU 的隐藏层维度
EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# 数据和网络文件路径
INP_FILE = 'Net3.inp'
DATASET_FILE = 'simulation_dataset.csv'


# --- 2. 图构建 ---
def create_graph_from_wntr(inp_file):
    """
    从 WNTR 的 .inp 文件创建 PyTorch Geometric 图对象。
    """
    print("正在从 .inp 文件构建管网图...")
    wn = wntr.network.WaterNetworkModel(inp_file)

    # 获取节点和管道信息
    nodes = list(wn.node_name_list)
    node_map = {name: i for i, name in enumerate(nodes)}

    # 创建边索引
    edge_sources = []
    edge_targets = []
    for pipe_name, pipe in wn.pipes():
        edge_sources.append(node_map[pipe.start_node_name])
        edge_targets.append(node_map[pipe.end_node_name])
        # GNN 通常处理无向图，所以添加反向边
        edge_sources.append(node_map[pipe.end_node_name])
        edge_targets.append(node_map[pipe.start_node_name])

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

    # 节点特征（这里我们只使用ID，实际应用中可加入高程、需求等）
    num_nodes = len(nodes)
    x = torch.eye(num_nodes)  # 使用 one-hot 编码作为初始特征

    graph = Data(x=x, edge_index=edge_index)
    print(f"图构建完成: {graph.num_nodes} 个节点, {graph.num_edges // 2} 条边。")
    return graph, nodes, node_map


# --- 3. 数据预处理 ---
def prepare_data(dataset_file, all_node_names, node_map, sensor_nodes):
    """
    加载数据集，进行标准化，并构建适用于时序预测的序列。
    """
    print("正在准备数据集...")
    df_raw = pd.read_csv(dataset_file, index_col=0)
    # 【【【 这是修复后的代码 (第1步) 】】】
    # 彻底解决索引问题：重置索引，让每一行都有一个唯一的整数索引 (0, 1, 2, ...)。
    # 之前重复的时间戳索引会被降级为一个普通列（我们之后不再需要它）。
    df = df_raw.reset_index(drop=True)

    # 确保列顺序与 node_map 一致
    df = df[all_node_names + ['simulation_id']]

    # 数据标准化
    scaler = MinMaxScaler()
    quality_data = df[all_node_names].values
    scaler.fit(quality_data)
    scaled_quality = scaler.transform(quality_data)

    # 构建时序样本
    X, y = [], []
    # 按 simulation_id 分组处理，防止跨模拟创建序列
    for sim_id, group in df.groupby('simulation_id'):
        # 【【【 这是修复后的代码 (第2步) 】】】
        # 'group.index' 现在是唯一的整数行号，可以直接用于索引。
        sim_data = scaled_quality[group.index]

        for i in range(len(sim_data) - LOOK_BACK):
            # 输入特征：仅使用传感器节点的数据
            sensor_indices = [node_map[name] for name in sensor_nodes]
            X.append(sim_data[i:i + LOOK_BACK, sensor_indices])
            # 目标：所有节点的下一时刻数据
            y.append(sim_data[i + LOOK_BACK, :])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    print(f"数据集准备完成: X shape: {X.shape}, y shape: {y.shape}")
    return X, y, scaler


# --- 4. GNN-GRU 模型定义 ---
class SpatioTemporalGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim_gnn, hidden_dim_gru, sensor_nodes_list):
        super(SpatioTemporalGNN, self).__init__()
        self.sensor_nodes_list = sensor_nodes_list
        self.gcn1 = GCNConv(in_channels, hidden_dim_gnn)
        self.gcn2 = GCNConv(hidden_dim_gnn, hidden_dim_gnn)

        # GRU的输入维度是 (传感器数量 * GNN隐藏维度)
        # 注意：这里我们做了一个简化，将GNN提取的特征直接用于GRU输入
        # 一个更复杂的模型可能会对每个节点的时序特征独立使用GRU
        self.gru = torch.nn.GRU(input_size=len(self.sensor_nodes_list) * hidden_dim_gnn,
                                hidden_size=hidden_dim_gru,
                                num_layers=1,
                                batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim_gru, out_channels)

    def forward(self, x_sequence, graph_data, node_map_ref):
        batch_size, seq_len, num_sensors = x_sequence.shape

        # GNN 部分：对图的静态特征进行编码
        # 在这个版本中，我们假设GNN用于学习一个全局的图嵌入
        # 简化处理：我们先不把时序数据直接送入GNN，而是用GNN学习节点间的关系
        # 注意：这是一个简化的基线模型，更高级的模型会将时序数据与GNN更紧密地结合
        node_embeddings = F.relu(self.gcn1(graph_data.x, graph_data.edge_index))
        node_embeddings = self.gcn2(node_embeddings, graph_data.edge_index)  # (num_nodes, hidden_dim_gnn)

        # 提取传感器节点的嵌入
        sensor_indices_tensor = torch.tensor([node_map_ref[name] for name in self.sensor_nodes_list], device=DEVICE)
        sensor_embeddings = node_embeddings[sensor_indices_tensor]  # (num_sensors, hidden_dim_gnn)

        # 将传感器时序数据与它们的图嵌入结合
        # (batch, seq_len, num_sensors) -> (batch, seq_len, num_sensors, 1)
        # (num_sensors, hidden_dim_gnn) -> (1, 1, num_sensors, hidden_dim_gnn)
        combined_features = x_sequence.unsqueeze(-1) * sensor_embeddings.unsqueeze(0).unsqueeze(0)

        # 准备GRU的输入
        # (batch, seq_len, num_sensors, hidden_dim_gnn) -> (batch, seq_len, num_sensors * hidden_dim_gnn)
        gru_input = combined_features.reshape(batch_size, seq_len, -1)

        # GRU 部分
        _, h_n = self.gru(gru_input)

        # 输出层
        out = self.linear(h_n.squeeze(0))
        return out


# --- 5. 训练与评估 ---
def train(model, loader, optimizer, loss_fn, graph, node_map_ref):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch, graph = x_batch.to(DEVICE), y_batch.to(DEVICE), graph.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x_batch, graph, node_map_ref)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, graph, node_map_ref):
    model.eval()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch, graph = x_batch.to(DEVICE), y_batch.to(DEVICE), graph.to(DEVICE)
        pred = model(x_batch, graph, node_map_ref)
        loss = loss_fn(pred, y_batch)
        total_loss += loss.item()

    return total_loss / len(loader)


# --- 6. 主程序 ---
if __name__ == "__main__":
    # 1. 构建图
    graph, all_node_names, node_map = create_graph_from_wntr(INP_FILE)

    # 2. 选择传感器节点（这是第二阶段要优化的）
    # 我们随机选择5个非水源、非水库的普通节点
    potential_sensors = [name for name in all_node_names if name not in ['River', 'Lake', '1', '2', '3']]
    sensor_nodes = random.sample(potential_sensors, 5)
    print(f"\n选定的传感器节点: {sensor_nodes}")

    # 3. 准备数据
    X, y, scaler = prepare_data(DATASET_FILE, all_node_names, node_map, sensor_nodes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # 4. 初始化模型、优化器和损失函数
    model = SpatioTemporalGNN(
        in_channels=graph.num_node_features,
        out_channels=graph.num_nodes,
        hidden_dim_gnn=HIDDEN_DIM_GNN,
        hidden_dim_gru=HIDDEN_DIM_GRU,
        sensor_nodes_list=sensor_nodes
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    print("\n--- 开始训练 ---")
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, loss_fn, graph, node_map)
        val_loss = evaluate(model, val_loader, loss_fn, graph, node_map)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")

    print("--- 训练完成 ---")

    # 5. 在测试集上评估并可视化
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss = evaluate(model, test_loader, loss_fn, graph, node_map)
    print(f"\n最终测试集均方误差 (MSE): {test_loss:.6f}")

    # 可视化对比
    print("正在生成预测结果对比图...")
    model.eval()
    x_sample, y_sample = next(iter(test_loader))
    x_sample, y_sample = x_sample.to(DEVICE), y_sample.to(DEVICE)
    graph = graph.to(DEVICE)

    with torch.no_grad():
        pred_sample = model(x_sample, graph, node_map)

    # 逆标准化
    y_sample_real = scaler.inverse_transform(y_sample.cpu().numpy())
    pred_sample_real = scaler.inverse_transform(pred_sample.cpu().numpy())

    # 选择几个非传感器的节点进行可视化
    non_sensor_nodes = [name for name in all_node_names if name not in sensor_nodes and name not in ['River', 'Lake']]
    plot_nodes = random.sample(non_sensor_nodes, 3)
    plot_indices = [node_map[name] for name in plot_nodes]

    fig, axes = plt.subplots(len(plot_nodes), 1, figsize=(12, 8), sharex=True)
    fig.suptitle('部分非监测点余氯浓度预测 vs. 真实值', fontsize=16)

    for i, node_idx in enumerate(plot_indices):
        ax = axes[i]
        ax.plot(y_sample_real[:50, node_idx], 'b-o', label='真实值', markersize=4)
        ax.plot(pred_sample_real[:50, node_idx], 'r--x', label='预测值', markersize=4)
        ax.set_title(f'节点 {plot_nodes[i]}')
        ax.set_ylabel('余氯 (mg/L)')
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('测试集样本点')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



