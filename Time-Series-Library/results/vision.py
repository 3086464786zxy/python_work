import numpy as np
import matplotlib.pyplot as plt
csv_path = "long_term_forecast_FECA_iTransformer_3_flour3_pl48_M_iTransformer_custom_ftM_sl16_ll4_pl4_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"
preds = np.load(csv_path + "\\pred.npy")
true = np.load(csv_path + "\\true.npy")
print("preds shape:", preds.shape)
# ===== 整体 MSE / MAE（所有样本 × 所有预测步）=====
diff = preds - true
mse = np.mean(diff ** 2)
mae = np.mean(np.abs(diff))
print(f"Overall MSE: {mse:.6f}")
print(f"Overall MAE: {mae:.6f}")
# ===== 可视化（仍然画 step=0）===== #
plt.figure()
plt.plot(preds[:, 1, 2])
plt.plot(true[:, 1, 2])
plt.legend(['preds', 'true'])
plt.title(f"Overall MSE={mse:.5f}, MAE={mae:.5f}")
plt.show()


#import numpy as np
#import matplotlib.pyplot as plt
#
## 读取数据
#
#csv_path = "long_term_forecast_noFECA_iTransformer_3_flour3_pl48_M_iTransformer_custom_ftM_sl16_ll4_pl4_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"
#preds = np.load(csv_path + "\\pred.npy")
#true  = np.load(csv_path + "\\true.npy")
#
## 计算 MSE 和 MAE
#diff = preds - true
#mse = np.mean(diff ** 2)
#mae = np.mean(np.abs(diff))
#
## 输出整体 MSE 和 MAE
#print(f"Overall MSE: {mse:.6f}")
#print(f"Overall MAE: {mae:.6f}")
#
## ===== 优化可视化：减少显示点数 / 增加x轴间隔 =====
#plt.figure(figsize=(10, 6))  # 调整图形大小
#
## 动态调整 step_range，确保不超出数据的最大索引
#step_range = np.arange(0, min(3500, preds.shape[0]), 20)  # 限制最大索引为 preds.shape[0]
#
#plt.plot(step_range, preds[step_range, 1, 1], label='preds', color='blue')
#plt.plot(step_range, true[step_range, 1, 1], label='true', color='orange')
#
## 增加标题和图例
#plt.legend(loc='best')
#plt.title(f"Overall MSE={mse:.5f}, MAE={mae:.5f}")
#
## 显示图形
#plt.show()

#import torch
#print("PyTorch版本:", torch.__version__)
#print("CUDA版本:", torch.version.cuda)
#print("CUDA是否可用:", torch.cuda.is_available())