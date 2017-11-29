import pytest
import numpy as np
import sys

sys.path.append("../")
from bayseg import BaySeg

@pytest.fixture
def bayseg_2d_4s():
    """Create a 2d 4-stamp """
    obs = np.ones((4, 4, 4))
    return BaySeg(obs, 3, stencil=4)


@pytest.fixture
def bayseg_2d_8s():
    obs = np.ones((4, 4, 4))
    return BaySeg(obs, 3, stencil=8)


def test_2d_gibbs_energy(bayseg_2d_4s):
    labels = np.array([[1., 1., 2., 1.],
                       [1., 1., 2., 1.],
                       [0., 0., 2., 0.],
                       [0., 0., 2., 0.]])

    beta = 1.

    sol = np.array([[[2., 3., 3., 2.],
                     [2., 3., 4., 2.],  # 0
                     [1., 2., 2., 2.],
                     [0., 1., 1., 1.]],

                    [[0., 1., 1., 1.],
                     [1., 2., 2., 2.],  # 1
                     [2., 3., 4., 2.],
                     [2., 3., 3., 2.]],

                    [[2., 2., 2., 1.],
                     [3., 3., 2., 2.],  # 2
                     [3., 3., 2., 2.],
                     [2., 2., 2., 1.]]])

    assert (bayseg_2d_4s._calc_gibbs_energy_vect(labels.flatten(), beta) == sol.reshape(16, 3)).all()


