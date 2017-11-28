import pytest
import numpy as np

import sys

sys.path.append("../")
from bayseg import pseudocolor


def test_pseudocolor_1d():
    """Assert correct pseudocoloring in 1d."""
    coords = np.array([np.arange(10)]).T

    sol = np.array([[0, 2, 4, 6, 8],
                    [1, 3, 5, 7, 9]]).T

    assert (pseudocolor(coords, None) == sol).all()


def test_pseudocolor_2d():
    """Assert correct pseudocoloring in 2d."""
    coords = np.array([[0, 0],
                       [0, 0],
                       [1, 1],
                       [1, 1],
                       [2, 2],
                       [2, 2],
                       [3, 3],
                       [3, 3],
                       [0, 1],
                       [2, 3],
                       [0, 1],
                       [2, 3],
                       [0, 1],
                       [2, 3],
                       [0, 1],
                       [2, 3]])

    extent = (4, 4)

    sol = np.array([[0,  2,  8, 10],
                    [1,  3,  9, 11],
                    [4,  6, 12, 14],
                    [5,  7, 13, 15]], dtype="int64")

    assert (sol == pseudocolor(coords, extent)).all()

