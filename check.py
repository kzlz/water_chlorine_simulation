import wntr
import sys

print("--- WNTR 模块信息 ---")
print(wntr)

print("\n--- WNTR 模块路径 (关键信息) ---")
try:
    print(wntr.__file__)
except AttributeError:
    print("无法找到 wntr.__file__ 属性，这非常不寻常！")

print("\n--- Python 搜索路径列表 ---")
# sys.path 显示了 Python 查找模块时会搜索的所有文件夹
for path in sys.path:
    print(path)