import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
df = pd.read_csv("df1_final_plot_data.csv")
df2 = pd.read_csv("C:/Users/User/Pictures/Deep-RL-Keras-master/Deep-RL-Keras-master/DDPG_static_car/df1.csv")
mean_data = df["Mean"]
mean_data2 = df2["Mean"]
standardized_mean_data = preprocessing.scale(mean_data)

def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y

fig, axes= plt.subplots(nrows=2, ncols=1,figsize=(8,6))
axes[0].plot(mean_data2, color='orange')
axes[0].plot(smooth(np.array(mean_data2)))
axes[1].plot(mean_data,color='orange')
axes[1].plot(smooth(np.array(mean_data)))
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Reward')
axes[0].set_title("Accumulated total reward for Scenario 1")
axes[1].set_title("Accumulated total reward for Scenario 2")
plt.tight_layout()

plt.show()
