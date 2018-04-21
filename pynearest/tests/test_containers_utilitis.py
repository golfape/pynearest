__author__ = 'golfape'

# Todo: include check of projection onto axis to verify that standard_normal really works in super-high dim

from pynearest.containers import _normalized, _unit_norm_random_vectors
import numpy as np
from nose.tools import assert_almost_equals


def test_normalize():
    A = np.array([[1.0, 2.0],[1.0,4.0]])
    print(A)
    A1 = _normalized(A,axis=-1)
    assert_almost_equals( (A1[0][0])**2 + (A1[0][1]**2), 1, places=4)


def test_high_dim():
    a = _unit_norm_random_vectors( 100, dim=100000 )


if __name__=="__main__":
    test_normalize()
    test_high_dim()