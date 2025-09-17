\# 基于 WNTR 的管网余氯模拟项目



\## 1. 项目简介



本项目使用 Python 的 `wntr` 库对 EPANET 的经典管网模型 \*\*Net3\*\* 进行二次开发，旨在模拟管网中余氯（Chlorine）的时空分布特性。



项目的主要功能是加载标准的管网文件，以编程方式将其从“水龄”模拟模式转换为“余氯”模拟模式，设置加氯源和衰减系数，最终运行模拟并以图表形式将结果可视化。



\## 2. 环境要求与安装



本项目在以下环境中开发和测试：

\* 操作系统: Windows

\* Python 版本: 3.11+

\* 主要依赖库:

&nbsp;   \* `wntr`

&nbsp;   \* `matplotlib`

&nbsp;   \* `pandas`

&nbsp;   \* `numpy`



您可以通过以下命令快速安装所有必要的依赖库：

```bash

pip install wntr matplotlib pandas numpy

