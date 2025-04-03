import matplotlib.pyplot as plt
import numpy as np

# 读取温度数据文件
data_file = 'T3_CUDA.data'
temperatures = []

with open(data_file, 'r') as file:
    for line in file:
        try:
            temp = float(line.strip())
            temperatures.append(temp)
        except ValueError:
            print(f"警告：无法解析行: {line}")

# 创建时间步数组
time_steps = np.arange(len(temperatures))

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
for i in range(0, len(temperatures), 200):
    if i > 0:  # 跳过第一个点
        plt.axvline(x=i, color='g', linestyle=':', alpha=0.5)
        plt.text(i+5, min(temperatures)+0.05, f'重新调整温度', 
                 rotation=90, verticalalignment='bottom', fontsize=8)

# 保存图像
plt.savefig('temperature_vs_time.png', dpi=300, bbox_inches='tight')
plt.savefig('temperature_vs_time.pdf', bbox_inches='tight')

# 显示图像
plt.show()

print("图像已保存为 'temperature_vs_time.png' 和 'temperature_vs_time.pdf'") 