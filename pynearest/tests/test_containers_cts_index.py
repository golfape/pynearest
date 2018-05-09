__author__ = 'golfape'

from pynearest.containers import ContinuousIndex
import numpy as np

def small_example():
    ci = ContinuousIndex( dim=4, num_basis_vectors=3, num_basis_collections=2 )
    ci.append( [0.1,  0.2, 1.0, 3.0]) # 0
    ci.append( [-0.1, 0.2, 1.0, 2.5])
    ci.append( [0.1,  0.3, 3.0, 1.1]) # 2
    ci.append( [0.13, 0.7, 3.0, 2.1])
    ci.append( [0.04, 0.4, 3.0, 2.2]) # 4
    ci.append( [-0.04,0.1, 1.0, -1.1])
    ci.append( [-0.04,0.1, 3.0, -1.1]) # 6 <--- closest
    ci.append( [-1.04,0.2, 3.0, 1.1])
    ci.append( [-2.04,0.3, 4.0, -1.1]) # 8
    ci.append( [-3.04,0.4, 1.0, 1.1])
    ci.append( [-4.04,0.5, 3.0, -1.1]) #10
    ci.append( [-5.04,0.6, 1.0, 5.1])
    ci.append( [-6.04,0.7, 1.0, -6.1]) #12
    ci.append( [-5.14,0.6, 2.0, 5.1])
    ci.append( [-6.24,0.7, 3.0, -6.1])
    ci.append( [-5.34,0.6, 1.0, 5.1])
    ci.append( [-6.44,0.7, 3.4, -6.1])
    ci.append( [-5.54,0.6, 3.5, 5.1])
    ci.append( [-6.64,0.7, 3.6, -6.1])
    ci.append( [-5.74,0.6, 3.7, 5.1])
    ci.append( [-6.84,0.7, 3.8, -6.1])
    ci.append( [-5.94,0.6, 3.9, 5.1])
    ci.append( [-6.94,0.7, 3.10, -6.1])
    ci.append( [-5.24,0.6, 3.11, 5.1])
    ci.append( [-6.24,0.7, 3.12, -6.1])
    return ci


def test_accuracy_small():
    """ See how often the correct data point is chosen """
    record = list()
    picks  = list()
    for _ in range(10):
        ci  = small_example()
        nbd = ci.getNeighbors( q=[-0.04,0.11,3.01,-1.11], k=1 )
        if len(nbd):
            pick = nbd[0][0]
        else:
            pick = -1
        picks.append(pick)
        record.append(int(pick==6))
    assert np.mean(record)>0.9,"Fraction correct fell below 90% "


def test_neighbours():
    """ TODO: Add correctness    """

    dim = 20
    num_query = 10
    ci  = ContinuousIndex( dim=dim, num_basis_vectors=10, num_basis_collections=3 )
    num_records_choices = [1,2,4,64,5000]
    for num_records in num_records_choices:
        for rec_no in range(num_records):
            v = np.random.rand(dim)
            ci.append( v )

        for _ in range(num_query):
            q = np.random.rand(dim)
            k = np.random.choice([1,2,4,16,64,128])
            k = min(k,len(ci))
            nbd = ci.getNeighbors(q=q, k=k, k0=None, k1=None, k2=None)
            assert len(nbd)<=k, "Didn't return enough neighbours"




if __name__=="__main__":
    test_neighbours()
    test_accuracy_small()