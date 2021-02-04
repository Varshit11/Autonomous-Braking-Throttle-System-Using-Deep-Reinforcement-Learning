import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
df = pd.read_csv("df1.csv")

mean_data = df["Mean"]
mean_data = mean_data[:2000]
standardized_mean_data = preprocessing.scale(mean_data)

def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y



plt.plot(mean_data, color='orange')
plt.plot(smooth(np.array(mean_data)))
plt.title("Train returns")
plt.xlabel("Episodes")
plt.ylabel("Average reward")
plt.show()

plt.plot(standardized_mean_data)
plt.plot(smooth(np.array(standardized_mean_data)))
plt.title("Train returns")
plt.show()

