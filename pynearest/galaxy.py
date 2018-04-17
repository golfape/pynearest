import numpy as np
from sortedcontainers import SortedDict
import math


def sd_ball( sd, key, k ):
    """ Return keys and indexes of neighbours in SortedDict
    :param key: float
    :param k:   Number of elements to include
    :param sd:  SortedDict
    :return:  [ (ky,val) ]   k-nearest neighbours
    """
    ndx      = sd.bisect( key )
    if ndx< len(sd)-k:
        ndx_low  = max( 0, ndx-k )
        ndx_high = min( len(sd)-1, ndx_low+2*k )
    else:
        ndx_low  = len(sd)-2*k-1
        ndx_high = len(sd)-1
    close = sorted( [ (sd.iloc[ i ],ndx-k+i) for i in range(ndx_low,ndx_high) ] )
    return close[:k]


class Galaxy( object ):

    # Stores vectors of length 'dim'
    # Retrieve nearest (approximately) with distance metric g

    def __init__( self, d, d1=0, M=25, L=3 ):
        """ Notation consistent with Li and Malik, "Fast k-Nearest Neighbour Search via Dynamic Continuous Indexing
            and the subsequent paper "Fast k-Nearest Neighbour Search via Prioritized DCI
        :param d:  Length of the part of vectors to be indexed
        :param d1: Length of the part of vectors to be stored but not indexed
        :param M:  Number of random vectors used in the indexing an retrieval
        :param L:  Number of collections of random vectors used in the indexing and retrieval
        """
        self.d   = d
        self.d1  = d1
        self.M   = M
        self.L   = L
        self.us  = [ [ np.random.randn(d) for _ in range(M) ] for l in range(L) ]
        self.T   = [ [ SortedDict() for _ in range(M)] for l in range(L) ]
        self.xs  = list()
        self.n   = 0


    def add( self, x ):
        self.xs.append( x )
        self.n = len(self.xs)
        if self.d1==0:
            xd = x
        else:
            xd = x[:self.d]
        for l in range(self.L):
            for m in range(self.M):
                xlm = np.dot( xd, self.us[l][m] )
                self.T[l][m][xlm] = self.n-1

    def nearest( self, q, k=20 ):
        """
        :param q:
        :return:
        """
        bs, xs = self.ball( q, k=k )
        ds     = [ np.sum( (q-x)**2 ) for x in xs ]
        ndx    = np.argmin( ds )
        return xs[ ndx ]

    def ball( self, q, k, k0=None, k1=None, d_prime=None ):
        """ Approximate k-nearest neighbour retrieval
        :param q:  Query point
        :param k0:  Number of neighbours to retrieve
        :param k1:  Number of points to visit in each composite index
        :param d_prime: Effective dimension
        :return: [ (i,x) ]
        """

        if self.d1==0:
            qd = q[:self.d]
        else:
            qd = q

        # Set search parameters
        d_prime = d_prime or 0.5*self.n
        assert( d_prime<=self.n )
        k0 = k0 or k* max(math.log(self.n/k),math.pow(1-self.n/d_prime))  # From Li and Malik
        k1 = k1 or k*self.M*max(math.log(self.n/k),math.pow(1-1.0/d_prime)) # From Li and Malik

        # Find nearby points (different for each choice of l)
        qlm    = [ [ np.dot( qd, self.us[l][m] ) for m in range(self.M) ] for l in range( self.L ) ]
        balls  = [ [ sd_ball(sd=self.T[l][m],key=qlm[l][m],k=k1) for m in range(self.M)] for l in range( self.L ) ]

        # For each l, a priority queues that contains indexes into the closest points
        P      = [ SortedDict() for _ in range(self.L) ]
        for l in range( self.L ):
            for m in range( self.M ):
                plm_, i   = balls[l][m][0]
                priority  = abs( plm_-qlm[l][m] )
                P[l][priority] = i

        # Use the proximity rankings in many different dimensions to construct an approximate neighbourhood
        C      = [ [ 0 for _ in range( self.n ) ] for _ in range(self.L) ]
        S      = [ set() for _ in range( self.L ) ]
        for j in range( k1-1 ):
            for l in range(self.L):
                if len(S[l])< k0:
                    pjl, hi = P[l].popitem(last=False)
                    self.T[l][j].popitem( last=False )
                    ky,val  = balls[l][j][0]
                    self.T[l][j][ val ]=ky
                    C[l][hi] = C[l][hi]+1
                    if C[l][hi]==self.m:
                        S[l].add(hi)

        i_candidates = set().union(*S)
        candidates   = sorted( [ (np.linalg.norm(self.xs[i][:self.d]-qd),i) for i in i_candidates ] )
        k_candidates = [ (i,self.xs[i]) for i in candidates[:k] ]
        return k_candidates






