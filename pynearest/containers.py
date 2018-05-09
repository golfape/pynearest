import numpy as np
import math
from sortedcontainers import SortedDict


class _WriteOnlyList( object ):

    """ The minimal set of methods needed for storing values in ContinuouslyIndexedDict """

    def __init__(self):
        self._list = list()

    def __getitem__(self, ndx):
        return self._list.__getitem__(ndx)

    def append(self, item):
        self._list.append(item)

    def __contains__(self, val):
        return val in self._list

    def index(self, item):
        return self._list.index(item)

    def __len__(self):
        return len(self._list)

    def len(self):
        return len(self._list)


class ContinuouslyIndexedDict( object ):

    """ A dictionary/queue style of object store where keys take values in R^d and values can be anything

        Keys support an approximate nearest neighbour search using a technique taken from Li and Malik,
           "Fast k-Nearest Neighbour Search via Prioritized DCI"
    """

    def __init__( self, key_dim, num_basis_vectors=25, num_basis_collections=3 ):
        """
        :param key_dim:     The dimension of the vectors that will be used as keys
        :param num_basis_vectors:    The parameter 'm' in the paper
        :param num_basis_collections:    The parameter 'L' in the paper
        :return:
        """
        assert isinstance( key_dim, int ), "Expecting dimension to be an integer"
        assert key_dim>0,"Expecting dimension to be greater than zero"

        self._key_dim = key_dim
        self._keys    = ContinuousIndex( dim= key_dim, num_basis_vectors=num_basis_vectors, num_basis_collections=num_basis_collections )
        self._values  = _WriteOnlyList()


    def append(self, key, val ):
        self._values.append( val )
        self._keys.append( key )

    def __delitem__(self, key):
        raise NotImplemented

    def len(self):
        return len( self._values )

    def __getitem__(self, key):
        loc = self._values.index( key )
        return self._values.__getitem__( loc )

    def getNeighbors( self, q, k, k0=None, k1=None, k2=None ):
        """ TODO: Better way to choose meta-params        """
        nbrs = self._keys.getNeighbors( q=q, k=k, k0=k0, k1=k1, k2=k2 )
        return [ (k, self._values[k]) for k,_ in nbrs]



class _SortedRealDict(SortedDict):
    """ A SortedDict in which keys are floats

    We need this for storing the projection value lookup tables. We could alternatively use a balanced tree
    Parent class docs: http://www.grantjenks.com/docs/sortedcontainers/sorteddict.html
    """

    # TODO: Prevent key collisions by overwriting the assignment functions

    def get_neighbor_loc(self, key, k):
        """ Return keys and indexes of neighbours in SortedDict
        :param key: float
        :param k:   Number of elements to include
        :param sd:  SortedDict
        :return:  [ (ky,val) ]   k-nearest neighbours
        """
        assert k<=len(self), "We expect k to not exceed the number of elements"
        ndx      = self.bisect_left( key )
        ndx_high = min(ndx+k+1,len(self)-1)
        ndx_low  = max(0,ndx-k-1)
        close = sorted( [ (np.linalg.norm(self.iloc[i]-key),(self.iloc[i],i)) for i in range(ndx_low, ndx_high+1) ] )
        neighbors = [ key_pos for dist,key_pos in close[:k]]
        return neighbors

    def get_neighbors(self, key, k):
        neighbors = self.get_neighbor_loc(key=key,k=k)
        return [ (ky,self.get( ky )) for ky,pos in neighbors ]


def _normalized(a, axis=-1, order=2):
    """ Normalize array in any dimension
    :param order:   The exponent in the norm
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def _unit_norm_random_vectors( num, dim, order=2 ):
    """ Generate random vectors with unit norm
    :param num:  Number of vectors
    :param dim:  Dimension of each vector
    :return:
    """
    xs = np.random.standard_normal( size=(num, dim) )
    return _normalized(xs,axis=-1,order=order)


def distance(q1,q2):
    return np.linalg.norm( np.asarray(q1)-np.asarray(q2))

def wiggle(x):
    for i in range(len(x)):
        x[i]=x[i]+1e-9*np.random.rand()
    return x

class ContinuousIndex( object ):

    """ Stores R^d vectors and permits retrieval of approximate k-nearest neighbours """

    def __init__( self, dim, num_basis_vectors=25, num_basis_collections=3 ):
        """ See Li and Malik,   Paper 1) "Fast k-Nearest Neighbour Search via Dynamic Continuous Indexing
            and the subsequent  Paper 2) "Fast k-Nearest Neighbour Search via Prioritized DCI

        :param dim:                     Length of vectors ('d' in the paper)
        :param num_basis_vectors:       Number of random vectors used in the indexing an retrieval ('M' in the paper)
        :param num_basis_collections:   Number of collections of random vectors used in the indexing and retrieval ('L' in the paper)
        """
        assert isinstance( num_basis_vectors, int ),     "Expecting num_basis_vectors to be an integer"
        assert num_basis_vectors>0,                      "Expecting num_basis_vectors to be greater than zero"
        assert isinstance( num_basis_collections, int ), "Expecting num_basis_collections to be an integer"
        assert num_basis_collections>0,                  "Expecting num_basis_collections to be greater than zero"

        self.dim   = dim
        self.num_basis_collections   = num_basis_collections # L
        self.num_basis_vectors       = num_basis_vectors     # M
        self.basis_vectors           = [    _unit_norm_random_vectors( num_basis_vectors, dim)       for _ in range(num_basis_collections) ]
        self.basis_inner_products    = [ [ _SortedRealDict() for _ in range(num_basis_vectors)] for _ in range(num_basis_collections) ]
        self.keys  = _WriteOnlyList()    # Monotonically growing list of keys taking values in R^d

    def __len__( self ):
        return len( self.keys )

    def __str__(self):
        return "ContinuousIndex: " + str(len(self)) + " vectors each of length " + str(self.dim)

    def __getitem__(self, ndx):
        return self.keys.__getitem__(ndx)

    def append( self, x):
        """ Append a key to the key store and insert its inner products in the lookups
        :param x:  [ float ]  New key in R^d to be appended
        :return:
        """
        num_keys = len( self )
        self.keys.append( x )
        for l in range(self.num_basis_collections):
            for m in range(self.num_basis_vectors):
                xlm = np.dot( x, self.basis_vectors[l][m] )
                while xlm in self.basis_inner_products[l][m]:
                    xlm = xlm+1e-10*np.random.rand(1)[0]
                    print("Wow, we had a key collision ... that's unlucky but it won't matter. Adding a tiny bit of noise. ")
                self.basis_inner_products[l][m][xlm] = num_keys
        return x

    def __delitem__(self, key):
        raise NotImplemented  # Likely to be inefficient


    def to_pickle(self, file_name):
        import pickle
        with open(file_name,'wb') as handle:
            pickle.dump( self, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def _getProjectionNeighbors(self, q, k2 ):
        """ For a query q, construct neighbourhoods around the projections and a priority queue for searching
        :param q:             [ float ]           Query vector
        :param k2:                                Number of neighbours around each projection
        :return:  qlm       [ [ float ] ]         Inner products of q against basis  [l][m]
                neighbors   [ [ [ (ky,val) ] ] ]  Lists of neighbors                 [l][m]
                priorities  [ _SortedRealDict ]   List of ordered dict in which keys are the difference between the projection of
                                                  a basis vector and a projection of a query vector, and values contain a tuple with the index into
                                                  the original data point and also the index into the basis (i,m)
        """
        # Compute inner products of the query vector q against the basis vectors in every collection
        qlm = [ [ np.dot( q, self.basis_vectors[l][m] ) for m in range(self.num_basis_vectors) ] for l in range( self.num_basis_collections ) ]

        # For each basis collection and each basis vector within, retrieve neighbours whose inner products against the same are closest to the inner product of q
        neighbors  = [ [ [] for m in range(self.num_basis_vectors)] for l in range( self.num_basis_collections ) ]
        for l in range( self.num_basis_collections ):
            for m in range( self.num_basis_vectors ):
                bip = self.basis_inner_products[l][m]
                qlm_ = qlm[l][m]
                ngh = bip.get_neighbors( key=qlm_,k=k2 )
                neighbors[l][m] = ngh

        # For each basis collection, order the basis vectors by how close the closest projection is. For each basis collection maintain this priority list (vals are pointers back to self.keys))
        # (This visitation pattern is the optimization distinguishing the first and second paper)
        priorities      = [ SortedDict() for _ in range(self.num_basis_collections) ]
        for l in range( self.num_basis_collections ):
            for m in range( self.num_basis_vectors ):
                plm_, i   = neighbors[l][m][0]    # Recall that i indexes back into the write only data store, i.e. self.keys[i]
                priority  = abs( plm_-qlm[l][m] )
                priorities[l][priority] = (i,m)                                           # We'll keep the dict sorted in ascending order

        return qlm, neighbors, priorities


    def _getWorstCaseParams(self,k,d_prime=None):
        """ Parameters from Li and Malik ... this function is not presently used
        :param k:
        :param k0:
        :param k1:
        :param d_prime:    Effective dimension
        :return:
          # These are overly conservative. The paper says "...it may be overly conservative in practice; so, these parameters may be chosen by cross-validation"
        """
        num_keys  = len(self)
        assert( d_prime<=num_keys )
        k0 = k*max(math.log(num_keys/k),math.pow(num_keys/k,1.0-num_keys/d_prime))
        k1 = k*self.num_basis_vectors*max(math.log(num_keys/k),math.pow(num_keys/k,1.0-1.0/d_prime))
        k0 = min(num_keys,int(k0+0.5))
        k1 = min(num_keys,int(k1+0.5))
        return k0,k1


    def _getPracticalParams(self,k,k0=None,k1=None,k2=None):
        """ Heuristic parameters for approximate nearest neighbor search
        :param k:    The number of neighbors
        :param k0:   The number of points to shortlist for each basis collection
        :param k1:   The number of points to shortlist
        :param k2:   The number of neighbors around each projection
        :return: k0, k1, k2
        """
        num_keys = len(self)
        k0 = k0 or 3*k
        assert k0>k,"Expecting k0>k"
        k1 = k1 or 10*k
        k0 = min(num_keys,int(k0+0.5))
        k1 = min(num_keys,int(k1+0.5))
        k2 = k2 or 200
        k2 = min(k2,len(self))
        k1 = min(k1,k2)
        k0 = min(k1,k0)
        return k0, k1,k2

    def _getShortlist(self, q, k0, k1, qlm, neighbors, priorities ):
        """ Create a list of points that might be close to the query vector q
        :params q           float                Query point
        :param k0           int                  Length of shortlist for each basis collection
        :param k1           int                  Length of combined shortlist
        :param qlm         [ [ float ] ]         Inner products of q against basis  [l][m]
        :param neighbors   [ [ [ (ky,val) ] ] ]  Lists of neighbors by basis vector  (indexed by [l][m] where l indexes the collection)
                priorities  [ _SortedRealDict ]   List of ordered dict in which keys are the difference between the projection of
                                                  a basis vector and a projection of a query vector, and values contain a tuple with the index into
                                                  the original data point and also the index into the basis (i,m)
        :return: [ (float,int) ]                 Distance to query and index of key
        """
        candidate_votes       = [ [ 0 for _ in range( len(self) ) ] for _ in range(self.num_basis_collections) ]
        shortlist_collections = [ set() for _ in range( self.num_basis_collections ) ]
        minimum_vote_count    = self.num_basis_vectors # This is assumed in the paper. The rationale is reasonable if you want to ensure there is no huge discrepancy in any dimension. However for some applications I suspect the equality need not be the best choice
        basis_vector_usage    = [ [ 0 for _ in range( self.num_basis_vectors ) ] for _ in range( self.num_basis_collections)]
        combined_shortlist    = list( set().union(*shortlist_collections) )

        sample_count = 0
        while len(combined_shortlist)<k1 and any( [len(shortlist_collections[l])<k0 for l in range(self.num_basis_collections)]) and any([len(pri) for pri in priorities]):
            for l in range(self.num_basis_collections):
                if len(shortlist_collections[l])< k0 and len(priorities[l])>0:
                    # Vote for the data point with the smallest error and remove from the l'th priority list
                    if len( priorities[l] ):
                        _, (i,_m) = priorities[l].popitem(last=False)
                        candidate_votes[l][i] += 1
                        basis_vector_usage[l][_m] += 1
                        if candidate_votes[l][i]==minimum_vote_count:
                            shortlist_collections[l].add(i)

                        # Add the next closest to priority list
                        used       = basis_vector_usage[l][_m]
                        if used<len(neighbors[l][_m]):
                            plm_, _i   = neighbors[l][_m][ used ]
                            _priority  = abs( plm_-qlm[l][_m] )
                            priorities[l][_priority] = (_i,_m)

            combined_shortlist = list( set().union(*shortlist_collections) )
            sample_count += 1

        if len(combined_shortlist)<k1:
            num_keys  = len(self)
            num_extra = k1-len(combined_shortlist)
            votes     = [ ( max( [ candidate_votes[l][i] for l in range( self.num_basis_collections) ] ),i) for i in range(num_keys) ]
            extras    = [ i for v,i in sorted(votes, reverse=True) ][:num_extra]
            combined_shortlist = list(combined_shortlist)+extras

        ordered_shortlist  = sorted( [ (distance(self.keys[i],q),i) for i in combined_shortlist ] )
        return ordered_shortlist

    def getNeighbors( self, q, k, k0=None, k1=None, k2=None ):
        """ Approximate k-nearest neighbour retrieval
        :param q:   Query point
        :params k:  Number of neighbours
        :param k0           int                  Length of shortlist for each basis collection
        :param k1           int                  Length of combined shortlist
        :param k2           int                  Length of neighbourhoods around basis vectors
        :return: [ (i,x) ]
        """

        k0, k1,k2 = self._getPracticalParams(k=k, k0=k0,k1=k1,k2=k2 )
        qlm, projection_neighbor_collections, priorities = self._getProjectionNeighbors(q=q, k2=k2)
        shortlist = self._getShortlist( q=q,k0=k0,k1=k1, qlm=qlm,neighbors=projection_neighbor_collections, priorities=priorities)
        neighbors =  [ (i,self.keys[i]) for dist,i in shortlist[:k] ]
        return neighbors








