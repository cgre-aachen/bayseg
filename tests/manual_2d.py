import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib

sys.path.append("../")
import bayseg

create_testing_data = importlib.import_module('create_testing_data')
observations, latent_2d = create_testing_data.create_2d_data(50, 50)

print(observations.shape)

plot = True

if plot:
    fig, ax = plt.subplots(ncols=5, figsize=(15, 7))
    for i in range(4 + 1):
        if i == 0:
            ax[i].imshow(latent_2d, interpolation="nearest", cmap=bayseg.cmap, norm=bayseg.cmap_norm)
            ax[i].set_ylabel("y")
            ax[i].set_title("Latent Field")
        else:
            ax[i].imshow(observations[:, :, i - 1], interpolation="nearest", cmap="gray")
            ax[i].set_title("Feature " + str(i))
        ax[i].grid(False)
        ax[i].set_xlabel("x")

    plt.show()
# **********************************************************************************************************

clf = bayseg.BaySeg(observations, 3, beta_init=1)

clf.fit(100, beta_jump_length=0.01, verbose=False)

clf.diagnostics_plot()

# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(latent_2d)
# ax[1].imshow(clf.labels[-1].reshape(clf.shape[0], clf.shape[1]))
# plt.show()
