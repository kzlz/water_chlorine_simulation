import wntr
import matplotlib.pyplot as plt
import matplotlib as mpl


def final_simulation():
    """
    此脚本为最终版本，请在强制重装 wntr 库后使用。
    """
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("警告：未找到中文字体'SimHei'，图表中的中文可能无法正常显示。")

    try:
        # 使用您手动修改并保存的 .net 文件，并指定'gbk'编码
        filename = '请在这里输入您保存的.net文件名'  # <-- 重要：请修改这一行
        wn = wntr.network.WaterNetworkModel(filename, encoding='gbk')

        # 使用现代wntr的标准API修改参数
        # 设置一个有意义的衰减系数
        wn.options.reaction.bulk_coeff = -0.3
        wn.options.reaction.wall_coeff = -0.1
        print("模型加载并配置成功。")
        print(f"全局本体衰减系数设置为: {wn.options.reaction.bulk_coeff}")
        print(f"全局管壁衰减系数设置为: {wn.options.reaction.wall_coeff}")

        # 运行模拟
        print("\n正在运行 EPANET 模拟...")
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        print("模拟完成。")

        # 分析与可视化
        if results.node['quality'] is not None:
            chlorine_results = results.node['quality']

            # 时间序列图
            plt.figure(figsize=(14, 7))
            time_in_hours = chlorine_results.index / 3600.0
            nodes_to_plot = ['103', '153', '215', '253', '111', '199', '231']

            # 使用更简洁的方式绘图
            chlorine_results[nodes_to_plot].plot(ax=plt.gca(), style=['-o', '-o', '-o', '-o', '--x', '--x', '--x'],
                                                 markersize=5)

            plt.title('部分节点余氯浓度随时间变化曲线', fontsize=16)
            plt.xlabel('时间 (小时)', fontsize=12)
            plt.ylabel('余氯浓度 (mg/L)', fontsize=12)
            plt.legend(title='节点')
            plt.grid(True)
            plt.ylim(bottom=0)
            plt.show()

            # 管网分布图
            snapshot_time_hr = 12
            # 确保快照时间在模拟结果的时间范围内
            snapshot_time_sec = min(snapshot_time_hr * 3600, chlorine_results.index[-1])
            chlorine_at_snapshot = chlorine_results.loc[snapshot_time_sec, :]

            fig, ax = plt.subplots(figsize=(18, 14))
            artists = wntr.graphics.plot_network(wn, node_attribute=chlorine_at_snapshot, node_size=25,
                                                 ax=ax, node_cmap=plt.get_cmap('viridis'),
                                                 title=f'T = {snapshot_time_hr} 小时 全管网余氯分布图')

            node_artist = artists['nodes']
            plt.colorbar(node_artist, ax=ax, label='余氯浓度 (mg/L)')
            plt.show()

    except Exception as e:
        print(f"发生了一个错误: {e}")


if __name__ == '__main__':
    final_simulation()