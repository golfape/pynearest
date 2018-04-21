__author__ = 'golfape'


from pynearest.containers import _WriteOnlyList
import numpy as np


def rnd_queue(n):
    n = 5000
    values = [ np.random.choice(['dog','cat','cow']) for _ in range(n) ]
    qu = _WriteOnlyList()
    for vl in values:
        qu.append(vl)
    return qu

def test_growing():
    qu = rnd_queue(n=5000)


def deep_cp(qu):
    qu_vals = [ v for v in qu]
    qu1 = _WriteOnlyList()
    for val in qu_vals:
        qu1.append(val)
    return qu1

def round_trip_property( qu ):
    len(qu)
    qu1 = deep_cp(qu)
    qu1.append('cat')
    assert len(qu1) == len(qu)+1

def index_property():
    qu = _WriteOnlyList()
    qu.append('dog')
    qu.append('cat')
    assert qu.index('cat')==1

def test_properties():
    num_tests = 10
    for test_num in range( num_tests ):
        qu = rnd_queue(n=np.random.choice([0,1,2,15,500]))
        round_trip_property(qu)
        index_property()

if __name__=="__main__":
    test_properties()
    test_growing()