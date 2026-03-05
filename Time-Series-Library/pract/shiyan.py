import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 目标函数
# ===============================
def func(x):
    # x shape: (n,2)
    return 3*np.cos(x[:,0]*x[:,1]) + 4 + x[:,0] + x[:,1]**2


# ===============================
# PSO参数
# ===============================
N = 100        # 粒子数
D = 2          # 维度
T = 200        # 最大迭代次数

c1 = 1.5
c2 = 1.5

w_max = 0.8
w_min = 0.4

x_min, x_max = -4, 4
v_min, v_max = -1, 1

# ===============================
# 初始化
# ===============================
pos = np.random.uniform(x_min, x_max, (N, D))
vel = np.random.uniform(v_min, v_max, (N, D))

pbest = pos.copy()
pbest_val = func(pos)

gbest_index = np.argmin(pbest_val)
gbest = pbest[gbest_index].copy()
gbest_val = pbest_val[gbest_index]

history = []

# ===============================
# 迭代
# ===============================
for t in range(T):

    w = w_max - (w_max - w_min) * t / T

    r1 = np.random.rand(N, D)
    r2 = np.random.rand(N, D)

    # 速度更新
    vel = (w * vel
           + c1 * r1 * (pbest - pos)
           + c2 * r2 * (gbest - pos))

    vel = np.clip(vel, v_min, v_max)

    # 位置更新
    pos = pos + vel
    pos = np.clip(pos, x_min, x_max)

    # 计算适应度
    fit = func(pos)

    # 更新个体最优
    mask = fit < pbest_val
    pbest[mask] = pos[mask]
    pbest_val[mask] = fit[mask]

    # 更新全局最优
    idx = np.argmin(pbest_val)
    if pbest_val[idx] < gbest_val:
        gbest_val = pbest_val[idx]
        gbest = pbest[idx].copy()

    history.append(gbest_val)

# ===============================
# 输出结果
# ===============================
print("最优解位置:", gbest)
print("最小函数值:", gbest_val)

# ===============================
# 收敛曲线
# ===============================
plt.plot(history)
plt.title("PSO Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best Value")
plt.grid()
plt.show()


# ===============================
# 画函数曲面 + 最优点
# ===============================
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)
Z = 3*np.cos(X*Y) + 4 + X + Y**2

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

ax.scatter(gbest[0], gbest[1], gbest_val,
           color='red', s=80, label="Best")

ax.set_title("Function Surface & PSO Minimum")
ax.legend()
plt.show()
