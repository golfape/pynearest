__author__ = 'golfape'

from ..galaxy import Galaxy
import numpy as np


def test_add():
    d  = 100
    n  = 500
    xs = [ np.random.randn(d) for _ in range(n) ]
    G  = Galaxy( d=d, d1=0, M=25, L=2 )
    for x in xs[:3]:
        G.add(x )





