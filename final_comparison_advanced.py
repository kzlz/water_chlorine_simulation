import wntr
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import random
import copy

# --- 1. 配置与超参数 ---
# (与之前基本相同)
INP_FILE = 'Net3.inp'
DATASET_FILE = 'simulation_dataset.csv'
FINAL_TRAINING_EPOCHS = 200  # 最大训练轮数
LEARNING_RATE = 0.001
BATCH_SIZE = 16
HIDDEN_DIM_GNN = 32
HIDDEN_DIM_GRU = 64
LOOK_BACK = 12
NUM_RANDOM_RUNS = 10
VALIDATION_SPLIT = 0.1  # 从训练数据中分出10%作为验证集
EARLY_STOPPING_PATIENCE = 15  # 如果验证损失连续15个epoch没有改善，则停止训练

# 我们找到的最优传感器布局
OPTIMAL_SENSORS = ['151', '601', '189', '201', '115']

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")


# --- 2. 早停机制类 ---
class EarlyStopping:
    """早停法以防止过拟合"""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model_state_dict = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model_state_dict = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


# --- 3. 模型与图构建 (与之前相同) ---
def build_graph(wn):
    G = wn.to_graph().to_undirected()
    node_map = {name: i for i, name in enumerate(wn.node_name_list)}
    valid_edges = [(u, v) for u, v in G.edges() if u in node_map and v in node_map]
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in valid_edges], dtype=torch.long).t().contiguous()
    return Data(edge_index=edge_index, num_nodes=len(node_map)), node_map


class GNN_GRU(torch.nn.Module):
    # (此部分代码与之前完全相同，此处省略以保持简洁)
    def __init__(self, num_nodes, num_sensors, node_map_sensor_indices, hidden_dim_gnn=32, hidden_dim_gru=64):
        super(GNN_GRU, self).__init__()
        self.num_nodes = num_nodes
        self.num_sensors = num_sensors
        self.node_map_sensor_indices = node_map_sensor_indices
        self.conv1 = GCNConv(1, hidden_dim_gnn)
        self.conv2 = GCNConv(hidden_dim_gnn, 1)
        self.gru = torch.nn.GRU(input_size=num_nodes, hidden_size=hidden_dim_gru, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim_gru, num_nodes)

    def forward(self, x_sequence, edge_index):
        batch_size, seq_len, _ = x_sequence.shape
        gru_inputs = []
        for t in range(seq_len):
            x_t = x_sequence[:, t, :]
            full_feature_matrix = torch.zeros(batch_size, self.num_nodes, 1, device=device)
            full_feature_matrix[:, self.node_map_sensor_indices, 0] = x_t
            x_gnn = full_feature_matrix.view(-1, 1)
            x_gnn = F.relu(self.conv1(x_gnn, edge_index))
            x_gnn = self.conv2(x_gnn, edge_index)
            x_gnn_reshaped = x_gnn.view(batch_size, self.num_nodes)
            gru_inputs.append(x_gnn_reshaped)
        gru_input_sequence = torch.stack(gru_inputs, dim=1)
        gru_out, _ = self.gru(gru_input_sequence)
        last_time_step_out = gru_out[:, -1, :]
        predictions = self.fc(last_time_step_out)
        return predictions


# --- 4. 数据准备 (返回scaler) ---
def prepare_data(df, all_node_names, sensor_nodes_to_use):
    scaler = MinMaxScaler(feature_range=(0, 1))
    # 注意：fit_transform 应该在整个数据集上进行，以保证归一化的一致性
    # 这里为了简化，我们假设每次调用的df都代表了全局的数据分布
    scaled_data = scaler.fit_transform(df[all_node_names].values)
    scaled_df = pd.DataFrame(scaled_data, columns=all_node_names)
    X, y = [], []
    df['temp_index'] = range(len(df))
    for sim_id, group in df.groupby('simulation_id'):
        group_start_index = group['temp_index'].iloc[0]
        group_end_index = group['temp_index'].iloc[-1]
        group_data = scaled_df.iloc[group_start_index: group_end_index + 1]
        for i in range(len(group_data) - LOOK_BACK):
            input_features = group_data[sensor_nodes_to_use].iloc[i:(i + LOOK_BACK)].values
            target_features = group_data.iloc[i + LOOK_BACK].values
            X.append(input_features)
            y.append(target_features)
    return np.array(X), np.array(y), scaler


# --- 5. 核心训练与评估函数 (已重构) ---
def train_and_evaluate_advanced(sensor_set, graph_data, node_map, X, y, scaler, all_node_names):
    print(f"\n--- [高级评估] 传感器布局: {sensor_set} ---")

    # 划分训练集和测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 从训练集中划分出验证集
    dataset_size = len(X_train_full)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_data = TensorDataset(torch.FloatTensor(X_train_full).to(device), torch.FloatTensor(y_train_full).to(device))

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    validation_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=valid_sampler)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test).to(device), torch.FloatTensor(y_test).to(device)),
                             batch_size=BATCH_SIZE)

    # 初始化模型
    sensor_indices_in_map = [node_map[name] for name in sensor_set]
    model = GNN_GRU(
        num_nodes=len(all_node_names),
        num_sensors=len(sensor_set),
        node_map_sensor_indices=sensor_indices_in_map
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=False)

    # 训练与验证循环
    for epoch in range(FINAL_TRAINING_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb, graph_data.edge_index.to(device))
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in validation_loader:
                pred = model(xb, graph_data.edge_index.to(device))
                val_loss = criterion(pred, yb)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        if (epoch + 1) % 20 == 0:
            print(
                f"    Epoch {epoch + 1:03d}/{FINAL_TRAINING_EPOCHS} | 训练损失: {loss.item():.6f} | 验证损失: {avg_val_loss:.6f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"    早停机制触发于 Epoch {epoch + 1}!")
            break

    # 加载性能最好的模型状态
    model.load_state_dict(early_stopping.best_model_state_dict)

    # 在测试集上进行最终评估
    model.eval()
    all_preds_normalized = []
    all_reals_normalized = []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb, graph_data.edge_index.to(device))
            all_preds_normalized.append(pred.cpu().numpy())
            all_reals_normalized.append(yb.cpu().numpy())

    preds_normalized = np.concatenate(all_preds_normalized)
    reals_normalized = np.concatenate(all_reals_normalized)

    # 【【【 关键步骤：反归一化 】】】
    preds_original = scaler.inverse_transform(preds_normalized)
    reals_original = scaler.inverse_transform(reals_normalized)

    # 计算多维度评估指标
    # 1. 在归一化尺度上计算
    mse_norm = mean_squared_error(reals_normalized, preds_normalized)
    r2_norm = r2_score(reals_normalized, preds_normalized)

    # 2. 在原始物理尺度上计算
    rmse_orig = np.sqrt(mean_squared_error(reals_original, preds_original))
    mae_orig = mean_absolute_error(reals_original, preds_original)

    print("--- 评估完成 ---")
    metrics = {
        'mse_normalized': mse_norm,
        'r2_score': r2_norm,
        'rmse_original (mg/L)': rmse_orig,
        'mae_original (mg/L)': mae_orig
    }
    for name, value in metrics.items():
        print(f"    {name}: {value:.6f}")

    return metrics


# --- 6. 主程序 ---
if __name__ == '__main__':
    # 加载管网和数据
    wn = wntr.network.WaterNetworkModel(INP_FILE)
    all_node_names = wn.node_name_list
    df = pd.read_csv(DATASET_FILE, index_col=0)
    df.reset_index(drop=True, inplace=True)
    graph_data, node_map = build_graph(wn)

    # 1. 评估最优布局
    X_optimal, y_optimal, scaler_optimal = prepare_data(df.copy(), all_node_names, OPTIMAL_SENSORS)
    metrics_optimal = train_and_evaluate_advanced(OPTIMAL_SENSORS, graph_data, node_map, X_optimal, y_optimal,
                                                  scaler_optimal, all_node_names)

    # 2. 评估10次随机布局
    nodes_to_exclude = {'River', 'Lake', '1', '2', '3'}
    candidate_nodes = [name for name in all_node_names if name not in nodes_to_exclude]

    random_metrics_list = []
    for i in range(NUM_RANDOM_RUNS):
        print(f"\n{'=' * 20} 随机实验轮次 {i + 1}/{NUM_RANDOM_RUNS} {'=' * 20}")
        random.seed(i)
        random_sensors = random.sample(candidate_nodes, 5)

        X_rand, y_rand, scaler_rand = prepare_data(df.copy(), all_node_names, random_sensors)
        metrics_rand = train_and_evaluate_advanced(random_sensors, graph_data, node_map, X_rand, y_rand, scaler_rand,
                                                   all_node_names)
        random_metrics_list.append(metrics_rand)

    # 3. 汇总和打印最终结果
    print("\n\n" + "=" * 80)
    print(" " * 28 + "最终科学对比结果 (高级版)")
    print("=" * 80)
    print(f"最优传感器布局: {OPTIMAL_SENSORS}")
    print(f"    - MSE (归一化):      {metrics_optimal['mse_normalized']:.6f}")
    print(f"    - R² Score:              {metrics_optimal['r2_score']:.6f}")
    print(f"    - RMSE (mg/L):           {metrics_optimal['rmse_original (mg/L)']:.6f}")
    print(f"    - MAE (mg/L):            {metrics_optimal['mae_original (mg/L)']:.6f}")

    print("\n随机传感器布局 (10次实验平均):")
    avg_random_metrics = {key: np.mean([m[key] for m in random_metrics_list]) for key in random_metrics_list[0]}
    std_random_metrics = {key: np.std([m[key] for m in random_metrics_list]) for key in random_metrics_list[0]}

    print(
        f"    - 平均 MSE (归一化):   {avg_random_metrics['mse_normalized']:.6f} (±{std_random_metrics['mse_normalized']:.6f})")
    print(
        f"    - 平均 R² Score:           {avg_random_metrics['r2_score']:.6f} (±{std_random_metrics['r2_score']:.6f})")
    print(
        f"    - 平均 RMSE (mg/L):        {avg_random_metrics['rmse_original (mg/L)']:.6f} (±{std_random_metrics['rmse_original (mg/L)']:.6f})")
    print(
        f"    - 平均 MAE (mg/L):         {avg_random_metrics['mae_original (mg/L)']:.6f} (±{std_random_metrics['mae_original (mg/L)']:.6f})")

    print("-" * 80)
    # 以最重要的、具有物理意义的 MAE 作为最终评判标准
    mae_improvement = (avg_random_metrics['mae_original (mg/L)'] - metrics_optimal['mae_original (mg/L)']) / \
                      avg_random_metrics['mae_original (mg/L)']
    print(f"相较于随机平均水平，平均绝对误差(MAE)降低: {mae_improvement:.2%}")
    print("=" * 80)
