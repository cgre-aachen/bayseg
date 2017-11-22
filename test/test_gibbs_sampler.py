import importlib
import sys
sys.path.append("../")
import bayseg
import numpy as np

# CREATE DATA
create_testing_data = importlib.import_module('create_testing_data')
coords, obs = create_testing_data.create_1d_data()

# INIT
clf = bayseg.BaySeg(coords, obs, 4, beta_init=100)

# FIT
clf.fit(50, beta_jump_length=5, verbose=False)

verbose = True

if verbose:
    print(np.shape(clf.labels))

# PLOT
clf.diagnostics_plot()

clf.plot_mu_stdev()
