基于GNN-GRU的供水管网余氯全域预测模型
1. 项目简介
本项目旨在解决一个核心的水务行业难题：如何利用少数监测点的数据，实时、准确地预测整个供水管网的余氯浓度分布。

我们首先使用 wntr 库对 EPANET 的经典管网模型 Net3 进行多次模拟，生成包含不同工况的余氯时空分布数据集。随后，我们构建了一个融合**图神经网络（GNN）和门控循环单元（GRU）**的深度学习模型，该模型能够：

利用 GNN 学习管网复杂的空间拓扑结构。

利用 GRU 捕捉水质动态变化的时间依赖性。

最终目标是训练一个能够仅根据少数几个传感器节点的数据，就能高精度推断出管网中所有节点余氯浓度的代理模型。

2. 项目结构
.
├── Net3.inp                # EPANET 原始管网模型文件
├── generate_dataset.py     # 脚本：运行多次模拟以生成数据集
├── train_gnn_gru.py        # 脚本：训练和评估 GNN-GRU 模型
├── simulation_dataset.csv  # 生成的模拟数据集 (运行 generate_dataset.py 后产生)
├── best_model.pth          # 训练好的最佳模型权重 (运行 train_gnn_gru.py 后产生)
└── README.md               # 本文档

3. 环境要求与安装
本项目在以下环境中开发和测试：

操作系统: Windows

Python 版本: 3.11+

主要依赖库:

wntr

pandas, numpy, scikit-learn

matplotlib

torch, torch_geometric

您可以通过以下命令快速安装所有必要的依赖库：

pip install wntr pandas numpy scikit-learn matplotlib torch torch_geometric

注意: torch_geometric 的安装可能依赖于您的 PyTorch 和 CUDA 版本。如果遇到问题，请参考其官方文档进行安装。

4. 使用方法
请按照以下步骤运行本项目：

第1步：生成模拟数据集
首先，我们需要运行多次模拟来创建一个包含多种工况的数据集。

python generate_dataset.py

该脚本会运行10次模拟（可在脚本内修改次数），并在项目根目录下生成 simulation_dataset.csv 文件。

第2步：训练并评估模型
数据集准备好后，运行主训练脚本：

python train_gnn_gru.py

该脚本会自动完成以下任务：

加载 Net3.inp 文件构建管网图结构。

加载 simulation_dataset.csv 并进行预处理。

随机选择5个节点作为“传感器”，用它们的历史数据作为输入。

训练 GNN-GRU 模型，目标是预测下一时刻所有节点的余氯浓度。

训练完成后，在测试集上评估模型性能，并保存最优模型为 best_model.pth。

生成并显示一张预测结果与真实值的对比图。

5. 当前进展
[已完成] 成功搭建并运行了 GNN-GRU 基线模型。

[已完成] 实现了完整的数据生成、预处理、模型训练和评估流程。

[下一步] 开展传感器最优布局研究，探索如何选择最有效的监测点组合以提升模型预测精度。
