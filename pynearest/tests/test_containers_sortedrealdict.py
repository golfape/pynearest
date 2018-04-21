__author__ = 'golfape'


from pynearest.containers import _SortedRealDict
import numpy as np


def deterministic_srd():
    srd  = _SortedRealDict()
    vals =  [(9.1,'nine'),(10.0,'ten'),(11.1,'eleven'),(12.2,'twelve'),(13.5,'thirteen'),(14.1,'forteen')]
    for v in vals:
       srd[v[0]]=v[1]
    return srd

def rnd_srd():
    srd  = _SortedRealDict()
    vals =  [(9.1,'nine'),(10.0,'ten'),(11.1,'eleven'),(12.2,'twelve'),(13.5,'thirteen'),(14.1,'forteen')]
    for v in vals:
        if np.random.choice(3)==1:
            srd[v[0]]=v[1]
    return srd

def test_del():
    srd = deterministic_srd()
    assert srd[9.1]=='nine'
    del srd[9.1]
    assert len(srd)==5


def test_index():
    srd= deterministic_srd()
    assert srd.index(11.1)==2

def test_construction():
    sd = deterministic_srd()
    assert sd.index(11.1)==2,"Indexing is not working"
    assert len(sd)==6, "len() isn't working"
    del sd[12.2]
    assert sd[9.1]=='nine',"Cannot find 9.1"
    assert len(sd)==5,"len() was not five"
    assert sd.index(11.1)==2,"Indexing is not working"

def test_getNeighbours():
    srd = deterministic_srd()
    nbd = srd.getNeighbors(key=15.,k=3)
    assert nbd[0][0]==14.1
    assert nbd[0][1]==5
    assert nbd[1][0]==13.5
    assert nbd[1][1]==4
    assert nbd[2][0]==12.2
    assert nbd[2][1]==3

    nbd = srd.getNeighbors(key=11.2,k=1)
    assert nbd[0][0]==11.1
    assert nbd[0][1]==2
    nbd = srd.getNeighbors(key=11.2,k=2)
    assert nbd[0][0]==11.1
    assert nbd[0][1]==2
    assert nbd[1][0]==12.2
    assert nbd[1][1]==3, "Wrong index"
    assert len(nbd)==2
    nbd = srd.getNeighbors(key=11.2,k=3)
    assert nbd[0][0]==11.1
    assert nbd[0][1]==2
    assert nbd[1][0]==12.2
    assert nbd[1][1]==3
    assert nbd[2][0]==10.0
    assert len(nbd)==3

    nbd = srd.getNeighbors(key=15,k=5)
    assert nbd[0][0]==14.1
    assert nbd[1][0]==13.5
    assert nbd[2][0]==12.2



if __name__=="__main__":
    test_getNeighbours()
    test_construction()
    test_del()