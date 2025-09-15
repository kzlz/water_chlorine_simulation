import wntr
import matplotlib.pyplot as plt
import matplotlib as mpl


def final_simulation():
    """
    此脚本加载 Net3.inp 文件，将其从水龄模拟模式修改为余氯模拟模式，
    然后运行模拟并可视化结果。
    """
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("警告：未找到中文字体'SimHei'，图表中的中文可能无法正常显示。")

    try:
        # --- 1. 文件设置 ---
        filename = 'Net3.inp'
        print(f"正在加载管网文件: {filename}...")

        # --- 2. 加载模型 ---
        wn = wntr.network.WaterNetworkModel(filename)
        print("管网加载成功。")

        # --- 3. 将模型配置为余氯模拟 ---
        print("\n正在将模型配置为余氯模拟模式...")
        wn.options.quality.parameter = 'CHLORINE'
        print(f"水质模拟参数已设置为: 'CHLORINE'")
        source_node = 'River'
        wn.add_source('Chlorine_Source_at_River', source_node, 'CONCEN', 1.2)
        print(f"在节点 '{source_node}' 添加了浓度为 1.2 mg/L 的加氯源。")
        wn.options.reaction.bulk_coeff = -0.3
        wn.options.reaction.wall_coeff = -0.1
        print(f"全局本体衰减系数设置为: {wn.options.reaction.bulk_coeff}")
        print(f"全局管壁衰减系数设置为: {wn.options.reaction.wall_coeff}")

        # --- 4. 运行模拟 ---
        print("\n正在运行 EPANET 模拟...")
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        print("模拟完成。")

        # --- 5. 分析与可视化 ---
        if results.node['quality'] is not None:
            chlorine_results = results.node['quality']

            # 时间序列图 (x轴优化版)
            plt.figure(figsize=(14, 7))
            time_in_hours = chlorine_results.index / 3600.0
            nodes_to_plot = ['103', '153', '215', '253', '111', '199', '231']
            for node_id in nodes_to_plot:
                style = '-o' if node_id in ['103', '153', '215', '253'] else '--x'
                plt.plot(time_in_hours, chlorine_results[node_id], style, markersize=5, label=f'节点 {node_id}')
            plt.title('部分节点余氯浓度随时间变化曲线', fontsize=16)
            plt.xlabel('时间 (小时)', fontsize=12)
            plt.ylabel('余氯浓度 (mg/L)', fontsize=12)
            plt.legend(title='节点')
            plt.grid(True)
            plt.xlim(0, 24)
            plt.ylim(bottom=0)
            plt.show()

            # 管网分布图 (颜色条修正版)
            snapshot_time_hr = 12
            snapshot_time_sec = min(snapshot_time_hr * 3600, chlorine_results.index[-1])
            chlorine_at_snapshot = chlorine_results.loc[snapshot_time_sec, :]

            fig, ax = plt.subplots(figsize=(18, 14))
            wntr.graphics.plot_network(wn, node_attribute=chlorine_at_snapshot, node_size=25,
                                       ax=ax, node_cmap=plt.get_cmap('viridis'),
                                       title=f'T = {snapshot_time_hr} 小时 全管网余氯分布图')

            # 【【【 这是修正后的颜色条代码 】】】
            # 创建一个独立的、与数据匹配的 mappable 对象用于颜色条
            norm = mpl.colors.Normalize(vmin=chlorine_at_snapshot.min(), vmax=chlorine_at_snapshot.max())
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=norm)
            sm.set_array([])  # 只需要 cmap 和 norm 信息

            # 从 figure 对象创建颜色条
            fig.colorbar(sm, ax=ax, label='余氯浓度 (mg/L)')
            plt.show()

    except Exception as e:
        print(f"发生了一个错误: {e}")


if __name__ == '__main__':
    final_simulation()