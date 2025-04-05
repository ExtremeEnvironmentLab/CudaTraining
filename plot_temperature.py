# 如果无法一键运行，可以通过以下命令运行：
# python .\plot_temperature.py

import matplotlib.pyplot as plt
import numpy as np

# --- 添加中文字体配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# --- 结束添加 ---


# 读取温度数据文件
data_file = 'T3_CUDA.data'
temperatures = []
pair_counts = [] # 添加用于存储近邻对数量的列表

with open(data_file, 'r') as file:
    for line in file:
        try:
            # --- 修改数据解析部分 ---
            parts = line.strip().split() # 按空格分割
            if len(parts) >= 1: # 确保至少有一个部分
                temp = float(parts[0]) # 第一个部分是温度
                temperatures.append(temp)
                if len(parts) >= 2: # 如果有第二个部分，解析为近邻对数量
                    try:
                        count = int(parts[1])
                        pair_counts.append(count)
                    except ValueError:
                        pair_counts.append(None) # 如果解析失败，添加None
                else:
                    pair_counts.append(None) # 如果没有第二个部分，添加None
            # --- 结束修改 ---
        except ValueError:
            print(f"警告：无法将 '{parts[0] if 'parts' in locals() and len(parts)>0 else line.strip()}' 解析为温度浮点数")
            temperatures.append(None) # 添加 None 以保持长度一致
            pair_counts.append(None)
        except Exception as e:
            print(f"处理行时发生意外错误: {line.strip()} - {e}")
            temperatures.append(None)
            pair_counts.append(None)

# 过滤掉读取失败的数据点 (确保时间和温度列表长度一致)
valid_indices = [i for i, t in enumerate(temperatures) if t is not None]
time_steps = np.array(valid_indices)
temperatures = [temperatures[i] for i in valid_indices]

# 注意：pair_counts 列表现在可能包含 None，如果需要绘制它，也需类似过滤

# 绘制温度随时间的变化图
plt.figure(figsize=(12, 6))
plt.plot(time_steps, temperatures, 'b-', linewidth=1.5)
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='目标温度 T=1.0')

# 添加标题和标签
plt.title('液态氩分子动力学模拟 - 温度随时间的变化', fontsize=16)
plt.xlabel('时间步', fontsize=14)
plt.ylabel('温度', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# 添加注释，标记温度重新调整的点
# 确保使用有效的时间步来定位垂直线
rescale_steps = [t for t in time_steps if t > 0 and t % 200 == 0]
for step in rescale_steps:
    plt.axvline(x=step, color='g', linestyle=':', alpha=0.5)
    # 找到温度列表中的最小值来定位文本，避免使用可能过滤掉的原始索引
    min_temp_in_view = min(temperatures) if temperatures else 0
    plt.text(step + 5, min_temp_in_view + 0.05, f'重新调整温度',
             rotation=90, verticalalignment='bottom', fontsize=8)

# 保存图像
plt.savefig('temperature_vs_time.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

print("图像已保存为 'temperature_vs_time.png' 和 'temperature_vs_time.pdf'") 