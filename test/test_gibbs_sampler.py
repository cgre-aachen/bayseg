import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import scipy.stats

import sys
sys.path.append("../")
import bayseg

# CREATE DATA
create_testing_data = importlib.import_module('create_testing_data')
coords, obs = create_testing_data.create_1d_data()

# INIT
clf = bayseg.BaySeg(coords, obs, 4, beta_init=10)

# FIT
clf.fit(25, beta_jump_length=5, verbose=False)

clf.diagnostics_plot()
