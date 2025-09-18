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
import time
import copy

# --- 1. 配置与超参数 ---
# [!!] 重要提示 [!!]
# 这是一个计算密集型任务。为了在合理时间内得到结果，我们将减少训练的轮数。
# 这足以比较不同传感器组合的优劣。
INP_FILE = 'Net3.inp'
DATASET_FILE = 'simulation_dataset.csv'
NUM_SENSORS_TO_FIND = 5  # 我们要寻找的最佳传感器数量
OPTIMIZATION_EPOCHS = 50  # 在优化过程中，每次评估只训练50轮以节省时间
LEARNING_RATE = 0.005
BATCH_SIZE = 16
HIDDEN_DIM_GNN = 32
HIDDEN_DIM_GRU = 64
LOOK_BACK = 12

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")


# --- 2. 模型定义 (与 train_gnn_gru.py 相同) ---
class GNN_GRU(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels_gnn, out_channels_gru):
        super(GNN_GRU, self).__init__()
        self.conv1 = GCNConv(in_channels, HIDDEN_DIM_GNN)
        self.conv2 = GCNConv(HIDDEN_DIM_GNN, out_channels_gnn)
        self.gru = torch.nn.GRU(input_size=out_channels_gnn,
                                hidden_size=out_channels_gru,
                                num_layers=2,
                                batch_first=True)
        self.fc = torch.nn.Linear(out_channels_gru, num_nodes)

    def forward(self, x_sequence, edge_index):
        batch_size, seq_len, num_sensor_nodes = x_sequence.shape

        # GNN部分处理每个时间步
        gnn_output_sequence = []
        for t in range(seq_len):
            x_t = x_sequence[:, t, :]
            # 为了通过GCN，需要一个完整的节点特征矩阵
            # 我们只在传感器位置有值，其他位置为0
            full_feature_matrix = torch.zeros(batch_size, self.num_nodes, 1, device=device)
            # 注意: 这里的映射需要正确的索引
            # x_t 的维度是 [batch_size, num_sensor_nodes]
            # 我们假设 x_sequence 的第三个维度已经按照传感器的真实节点ID排列
            # 在实际使用中，需要一个从 sensor_index 到 global_node_index 的映射

            # 这个实现过于复杂，我们简化一下逻辑：
            # GNN的目的是学习空间关系，我们可以先用GNN处理图，再用GRU处理时间
            # 但一个更简单且有效的方法是先用GRU捕捉每个传感器的时序动态
            # 然后用一个全连接层或者其他方式融合信息。
            # 为了保持与原baseline一致的结构，我们在此处不修改模型结构，
            # 假定输入x已经是某种形式的全图表示，这里我们直接传递给GRU
            pass  # 模型的结构保持不变，但在数据准备中会有调整

        # 我们直接使用GRU处理传感器数据，然后用FC扩展到全图
        # 这是一个简化但有效的模型
        # (我们将跳过GNN部分，因为在优化循环中，每次的输入图节点都不同，处理起来很复杂)
        # 为了简化优化过程，我们暂时使用一个纯GRU+FC的模型进行传感器选择
        # 这是一个常见的工程折衷

        # GRU 输入: (batch_size, seq_len, num_sensors)
        gru_out, _ = self.gru(x_sequence)

        # 我们只取最后一个时间步的输出
        last_time_step_out = gru_out[:, -1, :]

        # 全连接层扩展到所有节点
        predictions = self.fc(last_time_step_out)
        return predictions


# 为了优化，我们使用一个更简单的模型结构，这在传感器选择中更高效
class SensorSelectorModel(torch.nn.Module):
    def __init__(self, num_sensors, num_nodes):
        super(SensorSelectorModel, self).__init__()
        self.gru = torch.nn.GRU(input_size=num_sensors,
                                hidden_size=HIDDEN_DIM_GRU,
                                num_layers=2,
                                batch_first=True)
        self.fc = torch.nn.Linear(HIDDEN_DIM_GRU, num_nodes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_sensors)
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        return self.fc(last_out)


# --- 3. 数据准备函数 (稍作修改) ---
def prepare_data_for_sensors(df, all_node_names, sensor_nodes_to_use):
    num_nodes = len(all_node_names)
    sensor_indices = [all_node_names.index(node) for node in sensor_nodes_to_use]

    # 只选择传感器列和目标列（所有节点）
    sensor_data = df[sensor_nodes_to_use].values
    target_data = df[all_node_names].values

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    scaled_sensors = scaler_X.fit_transform(sensor_data)
    scaled_targets = scaler_y.fit_transform(target_data)

    X, y = [], []
    # 按照 simulation_id 分组处理
    df['temp_index'] = range(len(df))
    for sim_id, group in df.groupby('simulation_id'):
        group_sensors = scaled_sensors[group['temp_index'].values]
        group_targets = scaled_targets[group['temp_index'].values]

        for i in range(len(group) - LOOK_BACK):
            X.append(group_sensors[i:(i + LOOK_BACK)])
            y.append(group_targets[i + LOOK_BACK])

    return np.array(X), np.array(y), scaler_y


# --- 4. 核心评估函数 ---
def evaluate_sensor_set(sensor_set, df_full, all_node_names):
    """
    接收一个传感器节点列表，训练模型并返回其在验证集上的MSE。
    """
    print(f"    评估传感器组合: {sensor_set} ...")

    # 1. 准备数据
    X, y, scaler = prepare_data_for_sensors(df_full, all_node_names, sensor_set)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = TensorDataset(torch.FloatTensor(X_train).to(device), torch.FloatTensor(y_train).to(device))
    val_data = TensorDataset(torch.FloatTensor(X_val).to(device), torch.FloatTensor(y_val).to(device))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE)

    # 2. 初始化模型
    model = SensorSelectorModel(num_sensors=len(sensor_set), num_nodes=len(all_node_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    # 3. 训练循环
    val_loss = float('inf')
    for epoch in range(OPTIMIZATION_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # 4. 评估
        model.eval()
        with torch.no_grad():
            losses = []
            for xb, yb in val_loader:
                pred = model(xb)
                losses.append(criterion(pred, yb).item())
            val_loss = np.mean(losses)

    print(f"    组合 {sensor_set} 的最终验证 MSE: {val_loss:.6f}")
    return val_loss


# --- 5. 主程序：贪心搜索 ---
if __name__ == '__main__':
    # 加载数据和管网信息
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    node_names = wn.node_name_list

    # 移除水源和湖泊，因为它们通常是边界条件，不是常规监测点
    nodes_to_exclude = {'River', 'Lake', '1', '2', '3'}
    candidate_nodes = [name for name in node_names if name not in nodes_to_exclude]

    print("正在加载和预处理完整数据集...")
    df = pd.read_csv(DATASET_FILE, index_col=0)
    # 重置索引，因为原始索引是时间，存在重复
    df.reset_index(drop=True, inplace=True)
    print("数据集加载完毕。")

    # 开始贪心搜索
    selected_sensors = []

    start_total_time = time.time()

    for i in range(NUM_SENSORS_TO_FIND):
        print(f"\n--- 正在寻找第 {i + 1}/{NUM_SENSORS_TO_FIND} 个最佳传感器 ---")
        start_iter_time = time.time()

        best_new_sensor = None
        best_score = float('inf')

        # 遍历所有尚未被选中的候选节点
        for candidate_node in candidate_nodes:
            # 创建一个临时的传感器组合进行测试
            current_test_set = selected_sensors + [candidate_node]

            # 评估这个组合的性能
            score = evaluate_sensor_set(current_test_set, df.copy(), node_names)

            # 如果当前组合更好，则记录下来
            if score < best_score:
                best_score = score
                best_new_sensor = candidate_node

        # 将本轮找到的最佳传感器添加到最终列表中
        if best_new_sensor:
            selected_sensors.append(best_new_sensor)
            candidate_nodes.remove(best_new_sensor)  # 从候选列表中移除
            print(f"\n>>> 找到第 {i + 1} 个传感器: {best_new_sensor}，组合验证 MSE: {best_score:.6f}")
        else:
            print("错误：无法找到任何可以改进性能的传感器。")
            break

        end_iter_time = time.time()
        print(f"--- 第 {i + 1} 轮搜索耗时: {(end_iter_time - start_iter_time) / 60:.2f} 分钟 ---")

    end_total_time = time.time()

    print("\n\n" + "=" * 50)
    print("          传感器优化搜索完成！")
    print("=" * 50)
    print(f"总耗时: {(end_total_time - start_total_time) / 60:.2f} 分钟")
    print(f"找到的最佳 {NUM_SENSORS_TO_FIND} 个传感器布局为:")
    print(selected_sensors)
    print("=" * 50)
