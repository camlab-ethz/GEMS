"""
**********
Eigenstuff
**********

Functions for calculating eigenstuff of graphs.
"""

from scipy import sparse as sps
import numpy as np
from scipy.sparse import linalg as spla
from numpy import linalg as la

from scipy.sparse import issparse
from netcomp.linalg.matrices import _flat,_eps


######################
## Helper Functions ##
######################


def _eigs(M,which='SR',k=None):
    """ Helper function for getting eigenstuff.

    Parameters
    ----------
    M : matrix, numpy or scipy sparse
        The matrix for which we hope to get eigenstuff.
    which : string in {'SR','LR'}
        If 'SR', get eigenvalues with smallest real part. If 'LR', get largest.
    k : int
        Number of eigenvalues to return

    Returns
    -------
    evals, evecs : numpy arrays
        Eigenvalues and eigenvectors of matrix M, sorted in ascending or
        descending order, depending on 'which'.

    See Also
    --------
    numpy.linalg.eig
    scipy.sparse.eigs        
    """
    n,_ = M.shape
    if k is None:
        k = n
    if which not in ['LR','SR']:
        raise ValueError("which must be either 'LR' or 'SR'.")
    M = M.astype(float)
    if issparse(M) and k < n-1:
        evals,evecs = spla.eigs(M,k=k,which=which)
    else:
        try: M = M.todense()
        except: pass
        evals,evecs = la.eig(M)
        # sort dem eigenvalues
        inds = np.argsort(evals)
        if which == 'LR':
            inds = inds[::-1]
        else: pass
        inds = inds[:k]
        evals = evals[inds]
        evecs = np.matrix(evecs[:,inds])
    return np.real(evals),np.real(evecs)


#####################
##  Get Eigenstuff ##
#####################

def normalized_laplacian_eig(A,k=None):
    """Return the eigenstuff of the normalized Laplacian matrix of graph
    associated with adjacency matrix A.

    Calculates via eigenvalues if 

    K = D^(-1/2) A D^(-1/2)

    where `A` is the adjacency matrix and `D` is the diagonal matrix of
    node degrees. Since L = I - K, the eigenvalues and vectors of L can 
    be easily recovered.

    Parameters
    ----------
    A : NumPy matrix
        Adjacency matrix of a graph

    k : int, 0 < k < A.shape[0]-1
        The number of eigenvalues to grab. 

    Returns
    -------
    lap_evals : NumPy array
       Eigenvalues of L

    evecs : NumPy matrix
       Columns are the eigenvectors of L

    Notes
    -----
    This way of calculating the eigenvalues of the normalized graph laplacian is
    more numerically stable than simply forming the matrix L = I - K and doing
    numpy.linalg.eig on the result. This is because the eigenvalues of L are
    close to zero, whereas the eigenvalues of K are close to 1.

    References
    ----------

    See Also
    --------
    nx.laplacian_matrix
    nx.normalized_laplacian_matrix
    """
    n,m = A.shape
    ##
    ## TODO: implement checks on the adjacency matrix
    ##
    degs = _flat(A.sum(axis=1))
    # the below will break if
    inv_root_degs = [d**(-1/2) if d>_eps else 0 for d in degs]
    inv_rootD = sps.spdiags(inv_root_degs, [0], n, n, format='csr')
    # build normalized diffusion matrix
    K = inv_rootD*A*inv_rootD
    evals,evecs = _eigs(K,k=k,which='LR')
    lap_evals = 1-evals
    return np.real(lap_evals),np.real(evecs)
