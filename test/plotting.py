import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import scipy.stats
import pickle

import sys
sys.path.append("../")
import hmrf_gmm
#plt.style.use('bmh')  # seaborn-colorblind
# load pickle sample file
clf = pickle.load(open("../data/example.p", "rb"))

clf.diagnostics_plot()

# clf.plot_mu_stdev()