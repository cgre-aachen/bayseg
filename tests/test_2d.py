import pytest
import numpy as np
import sys
from .create_testing_data import create_2d_data

sys.path.append("../")
from bayseg import BaySeg

# create input data
data_2d, latent_2d = create_2d_data(30, 40)


@pytest.fixture
def bayseg_2d():
    """Creates a 2D BaySeg instance with the given input data."""
    return BaySeg(data_2d, 3)


def test_2d_features_vector(bayseg_2d):
    """Assert flattened feature space shape."""
    assert np.shape(bayseg_2d.feat) == (1200, 4)


