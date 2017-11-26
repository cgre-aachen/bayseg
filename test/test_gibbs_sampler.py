import importlib
import sys
sys.path.append("../")
import bayseg
import numpy as np

# CREATE DATA
create_testing_data = importlib.import_module('create_testing_data')
coords, obs, latent_1d = create_testing_data.create_1d_data()

# INIT
clf = bayseg.BaySeg(obs, 3, beta_init=1)

# FIT
clf.fit(50, beta_jump_length=5, verbose=False)

# *******************************************
verbose = True
plot = True
# *******************************************

if verbose:
    print("labels shp:", np.shape(clf.labels))

# PLOT

if plot:
    clf.diagnostics_plot(true_labels=latent_1d)
    # clf.plot_mu_stdev()
