__author__ = 'golfape'

from pynearest.containers import ContinuousIndex, _unit_norm_random_vectors
import numpy as np
import pickle
import timeit

import time
import os

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


def create_example( dim=100, num_records=10000, num_basis_vectors=10, num_basis_collections=3 ):
    ci  = ContinuousIndex( dim=dim, num_basis_vectors=num_basis_vectors, num_basis_collections=num_basis_collections )
    for rec_no in range(num_records):
        v = np.random.rand(dim)
        ci.append( v )
    return ci

def save_example(dim,num_records):
    ci = create_example( dim=dim, num_records=num_records )
    ci.to_pickle('pickled/example_'+str(dim)+'_by_'+str(num_records)+'.pkl')


def load_example(dim,num_records):
    fn = 'pickled/example_'+str(dim)+'_by_'+str(num_records)+'.pkl'
    if os.path.isfile(fn):
        try:
            with open(fn,'rb') as handle:
                ci = pickle.load(handle)
            return ci
        except EOFError:
            print(fn+" couldn't be loaded")

def load_or_create_example(dim,num_records):
    ci = load_example(dim=dim,num_records=num_records)
    if ci is None:
        print("Creating the pickled example from scratch...could take a while")
        save_example(dim=dim,num_records=num_records)
        ci = load_example(dim=dim,num_records=num_records)
    return ci

############################################################################################################


def study_matrix_performance():
    num_records = 1000*1000*10
    dim         = 100
    num_query   = 100
    A           = np.random.rand(num_records,dim)
    for _ in range(num_query):
        q           = np.random.rand(dim,1)
        v           = np.dot( A, q )

def study_norm_performance():
    num_records = 1000*1000
    dim         = 100
    num_query   = 100
    a           = np.random.rand(dim,1)
    b           = np.random.rand(dim,1)
    for _ in range(num_records):
        n = np.linalg.norm( a-b )


def study_get_timing(dim,num_records):
   with Timer() as t:
       ci = load_or_create_example(dim=dim,num_records=num_records)
   print('Loading data took %.03f sec.' % t.interval)
   num_query = 10
   dim = ci.dim
   for _ in range(num_query):
        q = np.random.rand(dim)
        with Timer() as t:
            nbd = ci.getNeighbors(q=q, k=5)
        print('Nearest neighbors retrieved in %.03f sec.' % t.interval)


if __name__=="__main__":
    study_get_timing( dim=100, num_records=100000 )