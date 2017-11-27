import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys

sys.path.append("../")
import bayseg

# picture dimensions
n = 30
m = 25

# **********************************************************************************************************
# latent field function


def func(x, y):
    return np.sin(y * x)


# **********************************************************************************************************
# latent field
xaxis = np.linspace(0, 4, n)
yaxis = np.linspace(0, 4, m)
x, y = np.meshgrid(xaxis, yaxis)
result = func(x, y)

latent_2d = np.zeros_like(result)
latent_2d[result > -0.5] = 1
latent_2d[result > 0.5] = 2

# **********************************************************************************************************
# sample
f = 4
c1 = multivariate_normal([7.9, 7.5, 11, 3], np.eye(f) * 0.35)
c2 = multivariate_normal([8, 7.9, 10.7, 2.9], np.eye(f) * 1.55)
c3 = multivariate_normal([8.5, 9, 9, 1], np.eye(f) * 0.95)

obs = np.zeros((m, n, f))

for i, x in enumerate(latent_2d[:, 0]):
    for j, y in enumerate(latent_2d[0, :]):
        # generate a 2-d random vector at each point
        if latent_2d[i, j] == 0:
            obs[i, j, :] = c1.rvs()
        elif latent_2d[i, j] == 1:
            obs[i, j, :] = c2.rvs()
        elif latent_2d[i, j] == 2:
            obs[i, j, :] = c3.rvs()

feature_vector = np.array(
    [obs[:, :, 0].flatten(), obs[:, :, 1].flatten(), obs[:, :, 2].flatten(), obs[:, :, 3].flatten()]).T

# **********************************************************************************************************
plot = True

if plot:
    fig, ax = plt.subplots(ncols=5, figsize=(15, 7))
    for i in range(f + 1):
        if i == 0:
            ax[i].imshow(latent_2d, interpolation="nearest", cmap=bayseg.cmap, norm=bayseg.cmap_norm)
            ax[i].set_ylabel("y")
            ax[i].set_title("Latent Field")
        else:
            ax[i].imshow(obs[:, :, i - 1], interpolation="nearest", cmap="gray")
            ax[i].set_title("Feature " + str(i))
        ax[i].grid(False)
        ax[i].set_xlabel("x")

    plt.show()
# **********************************************************************************************************

clf = bayseg.BaySeg(obs, 3, beta_init=10)

clf.fit(50, beta_jump_length=0.5)

clf.diagnostics_plot()
