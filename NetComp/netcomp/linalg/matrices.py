"""
********
Matrices
********

Matrices associated with graphs. Also contains linear algebraic helper functions.
"""

from scipy import sparse as sps
from scipy.sparse import issparse
import numpy as np

_eps = 10**(-10) # a small parameter

######################
## Helper Functions ##
######################


def _flat(D):
    """Flatten column or row matrices, as well as arrays."""
    if issparse(D):
        raise ValueError('Cannot flatten sparse matrix.')
    d_flat = np.array(D).flatten()
    return d_flat


def _pad(A,N):
    """Pad A so A.shape is (N,N)"""
    n,_ = A.shape
    if n>=N:
        return A
    else:
        if issparse(A):
            # thrown if we try to np.concatenate sparse matrices
            side = sps.csr_matrix((n,N-n))
            bottom = sps.csr_matrix((N-n,N))
            A_pad = sps.hstack([A,side])
            A_pad = sps.vstack([A_pad,bottom])
        else:
            side = np.zeros((n,N-n))
            bottom = np.zeros((N-n,N))
            A_pad = np.concatenate([A,side],axis=1)
            A_pad = np.concatenate([A_pad,bottom])
        return A_pad


########################
## Matrices of Graphs ##
########################


def degree_matrix(A):
    """Diagonal degree matrix of graph with adjacency matrix A

    Parameters
    ----------
    A : matrix
        Adjacency matrix

    Returns
    -------
    D : SciPy sparse matrix
        Diagonal matrix of degrees.
    """
    n,m = A.shape
    degs = _flat(A.sum(axis=1))
    D = sps.spdiags(degs,[0],n,n,format='csr')
    return D


def laplacian_matrix(A,normalized=False):
    """Diagonal degree matrix of graph with adjacency matrix A

    Parameters
    ----------
    A : matrix
        Adjacency matrix
    normalized : Bool, optional (default=False)
        If true, then normalized laplacian is returned.

    Returns
    -------
    L : SciPy sparse matrix
        Combinatorial laplacian matrix.
    """
    n,m = A.shape
    D = degree_matrix(A)
    L = D - A
    if normalized:
        degs = _flat(A.sum(axis=1))
        rootD = sps.spdiags(np.power(degs,-1/2), [0], n, n, format='csr')
        L = rootD*L*rootD
    return L
