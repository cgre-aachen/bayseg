import pytest
import numpy as np
import sys
import pandas as pd
sys.path.append("../")
from bayseg import BaySeg
import os

rs = 31258

f1_FILENAME = os.path.join(os.path.dirname(__file__), 'feature_1')
f2_FILENAME = os.path.join(os.path.dirname(__file__), 'feature_2')
comp_FILENAME = os.path.join(os.path.dirname(__file__), 'test_2d_data.npy')


# load features
f1 = pd.read_csv(f1_FILENAME, header=None).values
f2 = pd.read_csv(f2_FILENAME, header=None).values

# shape into dataset
observations = np.zeros((2, 100, 100))
observations[0, :, :] = f1
observations[1, :, :] = f2
obs = observations.T


@pytest.fixture
def bayseg_2d():
    """Creates a 2D BaySeg instance with the given input data."""
    np.random.seed(rs)
    clf = BaySeg(obs, 3, beta_init=0.1)
    return clf


def test_2d_run(bayseg_2d):
    np.random.seed(rs)
    bayseg_2d.fit(25, beta_jump_length=0.01)
    comp = np.load(comp_FILENAME)
    assert (bayseg_2d.labels[-1] == comp).all()


def test_2d_features_vector(bayseg_2d):
    """Assert flattened feature space shape."""
    assert np.shape(bayseg_2d.feat) == (10000, 2)



