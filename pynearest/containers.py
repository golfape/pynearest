import numpy as np
import math
from sortedcontainers import SortedDict

# TODO: Read http://ipython-books.github.io/45-understanding-the-internals-of-numpy-to-avoid-unnecessary-array-copying/ and then optimize this
# Apologies to my fellow expatriates for the spelling of neighbor

class _WriteOnlyList( object ):

    """ The minimal set of methods needed for storing values in ContinuouslyIndexedDict """

    def __init__(self):
        self._list = list()

    def __getitem__(self, ndx):
        return self._list.__getitem__(ndx)

    def append(self, item):
        self._list.append(item)

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

    def getNeighbors( self, q, k ):
        keys = self._keys.getNeighbors( q=q, num_neighbors=k )
        return [ (k, self._values[k]) for k in keys]



class _SortedRealDict(SortedDict):

    """ We need this for storing the projection value lookup tables. We could alternatively use a balanced tree  """

    # Parent class docs: http://www.grantjenks.com/docs/sortedcontainers/sorteddict.html

    # TODO: Ensure appending does not overwrite by accident

    def getNeighbors(self, key, k):
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
        return [v for k,v in close[:k]]


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
        self.keys  = _WriteOnlyList()        # Queue of keys taking values in R^d

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
        _n = len( self )
        self.keys.append( x )
        for l in range(self.num_basis_collections):
            for m in range(self.num_basis_vectors):
                xlm = np.dot( x, self.basis_vectors[l][m] )
                self.basis_inner_products[l][m][xlm] = _n

    def __delitem__(self, key):
        raise NotImplemented  # Likely to be inefficient


    def getNeighbors( self, q, num_neighbors, k0=None, k1=None, d_prime=None ):
        """ Approximate k-nearest neighbour retrieval
        :param q:   Query point
        :param k0:  Number of neighbours to retrieve prior to final selection
        :param k1:  Number of points to visit in each composite index
        :param d_prime: Effective dimension
        :return: [ (i,x) ]
        """


        # Set worst case tuning parameters
        num_keys  = len(self)
        d_prime = d_prime or 0.5*num_keys
        assert( d_prime<=num_keys )
        k0 = k0 or num_neighbors*max(math.log(num_keys/num_neighbors),math.pow(num_keys/num_neighbors,1.0-num_keys/d_prime))       # From Li and Malik
        k1 = k1 or num_neighbors*self.num_basis_vectors*max(math.log(num_keys/num_neighbors),math.pow(num_keys/num_neighbors,1.0-1.0/d_prime))   # From Li and Malik (very conservative)

        # These are overly conservative. The paper says "...it may be overly conservative in practice; so, these parameters may be chosen by cross-validation"
        # TODO: Find a better way to choose the parameters.
        # TODO: Short circuit all this if there are few keys in total
        k0 = min(k0, 2*num_neighbors+5)
        k1 = min(num_keys,2*k0+1)
        k0 = min(num_keys,int(k0+0.5))
        k1 = min(num_keys,int(k1+0.5))

        # Compute inner products of the query vector q against the basis vectors in every collection
        qlm    = [ [ np.dot( q, self.basis_vectors[l][m] ) for m in range(self.num_basis_vectors) ] for l in range( self.num_basis_collections ) ]

        # For each basis collection and each basis vector within, retrieve neighbours whose inner products against the same are closest to the inner product of q
        projection_neighbor_collections  = [ [ self.basis_inner_products[l][m].getNeighbors( key=qlm[l][m],k=k1 ) for m in range(self.num_basis_vectors)] for l in range( self.num_basis_collections ) ]

        # For each basis collection, order the basis vectors by how close the closest projection is. For each basis collection maintain this priority list (vals are pointers back to self.keys))
        # (This visitation pattern is the optimization distinguishing the first and second paper)
        priorities      = [ SortedDict() for _ in range(self.num_basis_collections) ]
        for l in range( self.num_basis_collections ):
            for m in range( self.num_basis_vectors ):
                plm_, i   = projection_neighbor_collections[l][m][0]    # Recall that i indexes back into the write only data store, i.e. self.keys[i]
                priority  = abs( plm_-qlm[l][m] )
                priorities[l][priority] = (i,m)                                           # We'll keep the dict sorted in ascending order

        # For each basis collection track votes for a data point
        candidate_votes       = [ [ 0 for _ in range( len(self) ) ] for _ in range(self.num_basis_collections) ]
        shortlist_collections = [ set() for _ in range( self.num_basis_collections ) ]
        minimum_vote_count    = self.num_basis_collections # This is assumed in the paper. The rationale is reasonable if you want to ensure there is no huge discrepancy in any dimension.
                                                           # However for some applications I suspect the equality need not be the best choice
        basis_vector_usage    = [ [ 0 for _ in range( self.num_basis_vectors ) ] for _ in range( self.num_basis_collections)]
        combined_shortlist    = list( set().union(*shortlist_collections) )

        loop_count = 0
        while len(combined_shortlist)<k1 and any([len(pri) for pri in priorities]) and loop_count<=1000:
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
                        if used<k1:
                            plm_, _i   = projection_neighbor_collections[l][_m][ used ]
                            _priority       = abs( plm_-qlm[l][_m] )
                            priorities[l][_priority] = (_i,_m)

            combined_shortlist = list( set().union(*shortlist_collections) )
            loop_count += 1

        ordered_shortlist  = sorted( [ (distance(self.keys[i],q),i) for i in combined_shortlist ] )
        neighbors =  [ (i,self.keys[i]) for dist,i in ordered_shortlist[:num_neighbors] ]
        return neighbors






