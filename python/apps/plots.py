
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Intel One Mono"
plt.rcParams["font.size"] = 12


import numpy as np
import pandas as pd









n = 200
D = np.zeros((n, 2))
D[:,0] = np.linspace(0, 10, n) + np.random.randn(n)
D[:,1] = D[:,0]**2 + np.random.randn(n) *10



plt.plot(kind="bar")
plt.title("template plot")
plt.ylabel("y label")
plt.xlabel("x label")
plt.plot(D[:, 0], D[:, 1], 'o')
# plt.show()


df = pd.DataFrame(data=D, columns=["var1", "var2"])
# sns.jointplot(x=df.columns[0], y=df.columns[1], data=df,)
plt.show()



