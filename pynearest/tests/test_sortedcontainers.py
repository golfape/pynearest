from sortedcontainers import SortedList, SortedDict
import numpy as np

# It is useful to have some tests here of the main project dependency TODO: make them better

def test_construction():
    sl = SortedList(range(100000))

def deterministic_sorteddict():
    sd = SortedDict()
    vals =  [(9.1,'nine'),(10.0,'ten'),(11.1,'eleven'),(12.2,'twelve'),(13.5,'thirteen'),(14.1,'forteen')]
    for k,v in vals:
            sd[k]=v
    return sd


def rnd_sorteddict():
    sd = SortedDict()
    vals =  [(9.1,'nine'),(10.0,'ten'),(11.1,'eleven'),(12.2,'twelve'),(13.5,'thirteen'),(14.1,'forteen')]
    for k,v in vals:
        if np.random.choice(3)==1:
            sd[k]=v
    return sd

def test_construction():
    sd = deterministic_sorteddict()
    assert len(sd)==6
    del sd[12.2]
    assert sd[9.1]=='nine'
    assert len(sd)==5
    assert sd.index(11.1)==2
    rsd = rnd_sorteddict()

if __name__=='__main__':
    test_construction()