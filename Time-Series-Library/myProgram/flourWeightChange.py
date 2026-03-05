import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ 读取数据
df = pd.read_csv('../data/Flour/flour.csv')

# 2️⃣ 确保日期是时间格式
df['date'] = pd.to_datetime(df['date'])

# 3️⃣ 按时间排序（防止乱序）
df = df.sort_values('date')

# 4️⃣ 绘制面粉重量随时间变化曲线
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['weight'], color='blue', linewidth=2)

plt.title('Flour Weight Over Time (面粉重量随时间变化曲线)', fontsize=14)
plt.xlabel('Date (日期)', fontsize=12)
plt.ylabel('Weight (重量)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
