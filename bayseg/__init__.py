"""Spatial segmentation with multiple features using Hidden Markov Random Fields and Finite Mixture Models

Approach based on Wang et al. 2016 paper

@author: Alexander Schaaf, Hui Wang
"""
import sys

from .bayseg import *

assert sys.version_info[0] >= 3, "HMRFGMM requires Python v3.X"  # sys.version_info[1] for minor e.g. 6
