"""
***********************
Fast Belief Propagation
***********************

The fast approximation of the Belief Propogation matrix.
"""

from scipy import sparse as sps
import numpy as np
from numpy import linalg as la

def fast_bp(A,eps=None):
    """Return the fast belief propogation matrix of graph associated with A.

    Parameters
    ----------
    A : NumPy matrix or Scipy sparse matrix
        Adjacency matrix of a graph. If sparse, can be any format; CSC or CSR
        recommended.

    eps : float, optional (default=None)
        Small parameter used in calculation of matrix. If not provided, it is
        set to 1/(1+d_max) where d_max is the maximum degree.

    Returns
    -------
    S : NumPy matrix or Scipy sparse matrix
        The fast belief propogation matrix. If input is sparse, will be returned
        as (sparse) CSC matrix.

    Notes
    -----

    References
    ----------

    """
    n,m = A.shape
    ##
    ## TODO: implement checks on the adjacency matrix
    ##
    degs = np.array(A.sum(axis=1)).flatten()
    if eps is None:
        eps = 1/(1+max(degs))
    I = sps.identity(n)
    D = sps.dia_matrix((degs,[0]),shape=(n,n))
    # form inverse of S and invert (slow!)
    Sinv = I + eps**2*D - eps*A
    try:
        S = la.inv(Sinv)
    except:
        Sinv = sps.csc_matrix(Sinv)
        S = sps.linalg.inv(Sinv)
    return S
