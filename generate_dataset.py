import wntr
import pandas as pd
import numpy as np
import time


def generate_dataset(num_simulations=10):
    """
    运行多次模拟以生成一个多样化的数据集。
    每次模拟使用随机的源头加氯浓度。
    """
    print(f"--- 开始生成数据集：总共将运行 {num_simulations} 次模拟 ---")

    filename = 'Net3.inp'
    all_results = []

    start_time = time.time()

    for i in range(num_simulations):
        print(f"\n--- 正在运行模拟 {i + 1}/{num_simulations} ---")

        try:
            # 1. 加载模型
            wn = wntr.network.WaterNetworkModel(filename)

            # 2. 配置模型
            wn.options.quality.parameter = 'CHLORINE'

            # 随机化初始条件：让源头浓度在 1.0 和 1.5 之间随机变化
            source_node = 'River'
            random_concentration = round(np.random.uniform(1.0, 1.5), 2)
            wn.add_source('Chlorine_Source', source_node, 'CONCEN', random_concentration)
            print(f"本次模拟源头浓度设置为: {random_concentration} mg/L")

            wn.options.reaction.bulk_coeff = -0.3
            wn.options.reaction.wall_coeff = -0.1

            # 3. 运行模拟
            sim = wntr.sim.EpanetSimulator(wn)
            results = sim.run_sim()

            # 4. 收集结果
            if results.node['quality'] is not None:
                # 给结果增加一列，用于区分是哪次模拟
                quality_df = results.node['quality']
                quality_df['simulation_id'] = i
                all_results.append(quality_df)
                print(f"模拟 {i + 1} 成功，已收集结果。")

        except Exception as e:
            print(f"模拟 {i + 1} 失败，错误: {e}")

    end_time = time.time()
    print(f"\n--- 所有模拟完成，总耗时: {end_time - start_time:.2f} 秒 ---")

    # 5. 合并并保存数据集
    if all_results:
        # 将所有模拟结果的DataFrame合并成一个大的DataFrame
        final_dataset = pd.concat(all_results)

        # 保存到CSV文件，方便下一步使用
        output_filename = 'simulation_dataset.csv'
        final_dataset.to_csv(output_filename)
        print(f"数据集已成功保存到文件: {output_filename}")
        print(f"总共生成了 {len(final_dataset)} 条数据记录。")
    else:
        print("未能成功生成任何模拟结果。")


if __name__ == '__main__':
    # 您可以修改这里的数字来决定要运行多少次模拟
    generate_dataset(num_simulations=10)