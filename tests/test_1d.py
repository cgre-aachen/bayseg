import pytest
import numpy as np
import sys
from .create_testing_data import create_1d_data

sys.path.append("../")
from bayseg import BaySeg

# create input data
data_1d, latent_1d = create_1d_data()


@pytest.fixture
def bayseg_1d():
    """Create a 1d BaySeg instance with the input data and 3 labels."""
    return BaySeg(data_1d, 3)


def test_1d_shape(bayseg_1d):
    """Test the input data shape."""
    assert bayseg_1d.shape == (500, 4)


def test_1d_n_feat(bayseg_1d):
    """Test the number of features."""
    assert bayseg_1d.n_feat == 4


def test_1d_dim(bayseg_1d):
    """Test the physical dimension of the instance."""
    assert bayseg_1d.dim == 1


def test_1d_features_vector(bayseg_1d):
    """Test the shape of the features vector."""
    assert np.shape(bayseg_1d.feat) == (500, 4)


def test_1d_pseudocolors(bayseg_1d):
    """Test the pseudocoloring for 1d data."""
    assert (bayseg_1d.colors == np.array([np.arange(0, 500, step=2), np.arange(1, 500, step=2)]).T).all()


def test_1d_gibbs_energy(bayseg_1d):
    """Test gibbs energy calculation in 1d."""
    labels = np.array([ 0, 1, 2, 2, 2, 1])
    result = np.array([[1, 1, 2, 2, 2, 1],
                       [0, 2, 1, 2, 1, 1],
                       [1, 1, 1, 0, 1, 0]]).T.astype(float)
    beta = 1.
    assert (bayseg_1d._calc_gibbs_energy_vect(labels, beta) == result).all()
