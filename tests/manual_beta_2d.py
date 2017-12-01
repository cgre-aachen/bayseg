import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import bayseg

# load features
f1 = pd.read_csv("../data/2d_anisotropic_mrf/feature_1", header=None).values
f2 = pd.read_csv("../data/2d_anisotropic_mrf/feature_2", header=None).values
# load reality
lf = pd.read_csv("../data/2d_anisotropic_mrf/latent_field", header=None).values

# shape into dataset
observations = np.zeros((2, 100, 100))
observations[0, :, :] = f1
observations[1, :, :] = f2
obs = observations.T

# instantiate classifier
clf = bayseg.BaySeg(obs, 3, beta_init=0.1)

# fit
clf.fit(100, beta_jump_length=0.01)

clf.diagnostics_plot()
clf.plot_mu_stdev()