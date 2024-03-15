"""
**********
Resistance
**********

Resistance matrix. Renormalized version, as well as conductance and commute matrices.
"""

import networkx as nx
from numpy import linalg as la
from scipy import linalg as spla
import numpy as np
from scipy.sparse import issparse

from netcomp.linalg.matrices import laplacian_matrix
from netcomp.exception import UndefinedException


def resistance_matrix(A,check_connected=True):
    """Return the resistance matrix of G.

    Parameters
    ----------
    A : NumPy matrix or SciPy sparse matrix
        Adjacency matrix of a graph.

    check_connected : Boolean, optional (default=True)
        If false, then the resistance matrix will be computed even for
        disconnected matrices. See Notes.

    Returns
    -------
    R : NumPy matrix
       Matrix of pairwise resistances between nodes.

    Notes
    -----
    Uses formula for resistance matrix R in terms of Moore-Penrose of
    pseudoinverse (non-normalized) graph Laplacian. See e.g. Theorem 2.1 in [1]. 

    This formula can be computed even for disconnected graphs, although the
    interpretation in this case is unclear. Thus, the usage of
    check_connected=False is recommended only to reduce computation time in a
    scenario in which the user is confident the graph in question is, in fact,
    connected.

    Since we do not expect the pseudoinverse of the laplacian to be sparse, we
    convert L to dense form before running np.linalg.pinv(). The returned
    resistance matrix is dense.

    See Also
    --------
    nx.laplacian_matrix

    References
    ----------
    .. [1] W. Ellens, et al. (2011)
       Effective graph resistance.
       Linear Algebra and its Applications, 435 (2011)

    """
    n,m = A.shape
    # check if graph is connected
    if check_connected:
        if issparse(A):
            G = nx.from_scipy_sparse_matrix(A)
        else:
            G = nx.from_numpy_matrix(A)
        if not nx.is_connected(G):
            raise UndefinedException('Graph is not connected. '
                                     'Resistance matrix is undefined.')
    L = laplacian_matrix(A)
    try: L = L.todense()
    except: pass
    M = la.pinv(L)
    # calculate R in terms of M
    d = np.reshape(np.diag(M),(n,1))
    ones = np.ones((n,1))
    R = np.dot(d,ones.T) + np.dot(ones,d.T) - M - M.T
    return R

def commute_matrix(A):
    """Return the commute matrix of the graph associated with adj. matrix A.

    Parameters
    ----------
    A : NumPy matrix or SciPy sparse matrix
        Adjacency matrix of a graph.

    Returns
    -------
    C : NumPy matrix
       Matrix of pairwise resistances between nodes.

    Notes
    -----
    Uses formula for commute time matrix in terms of resistance matrix, 

    C = R*2*|E|

    where |E| is the number of edges in G. See e.g. Theorem 2.8 in [1].

    See Also
    --------
    laplacian_matrix
    resistance_matrix

    References
    ----------
    .. [1] W. Ellens, et al. (2011)
       Effective graph resistance.
       Linear Algebra and its Applications, 435 (2011)

    """
    R = resistance_matrix(A)
    E = A.sum()/2 # number of edges in graph
    C = 2*E*R
    return C

def renormalized_res_mat(A,beta=1):
    """Return the renormalized resistance matrix of graph associated with A.

    To renormalize a resistance R, we apply the function

    R' = R / (R + beta)

    In this way, the renormalized resistance of nodes in disconnected components
    is 1. The parameter beta determines the penalty for disconnection. If we set
    beta to be approximately the maximum resistance found in the network, then
    the penalty for disconnection is at least 1/2.

    Parameters
    ----------
    A : NumPy matrix or SciPy sparse matrix
        Adjacency matrix of a graph.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist. If
       nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    beta : float, optional
       Scaling parameter in renormalization. Must be greater than or equal to
       1. Determines how heavily disconnection is penalized.

    Returns
    -------
    R :  NumPy array
       Matrix of pairwise renormalized resistances between nodes.

    Notes
    -----
    This function converts to a NetworkX graph, as it uses the algorithms
    therein for identifying connected components.

    See Also
    --------
    resistance_matrix

    """
    if issparse(A):
        G = nx.from_scipy_sparse_matrix(A)        
    else:
        G = nx.from_numpy_matrix(A)
    n = len(G)
    subgraphR = []
    for subgraph in nx.connected_component_subgraphs(G):
        a_sub = nx.adjacency_matrix(subgraph)
        r_sub = resistance_matrix(a_sub)
        subgraphR.append(r_sub)
    R = spla.block_diag(*subgraphR)
    # now, resort R so that it matches the original node list
    component_order = []
    for component in nx.connected_components(G):
        component_order += list(component)
    component_order = list(np.argsort(component_order))
    R = R[component_order,:]
    R = R[:,component_order]
    renorm = np.vectorize(lambda r: r/(r+beta))
    R = renorm(R)
    # set resistance for different components to 1
    R[R==0]=1
    R = R - np.eye(n) # don't want diagonal to be 1
    return R


def conductance_matrix(A):
    """Return the conductance matrix of G.

    The conductance matrix of G is the element-wise inverse of the resistance
    matrix. The diagonal is set to 0, although it is formally infinite. Nodes in
    disconnected components have 0 conductance.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist. If
       nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    C :  NumPy array
       Matrix of pairwise conductances between nodes.


    See Also
    --------
    resistance_matrix
    renormalized_res_mat

    """
    if issparse(A):
        G = nx.from_scipy_sparse_matrix(A)        
    else:
        G = nx.from_numpy_matrix(A)
    subgraphC = []
    for subgraph in nx.connected_component_subgraphs(G):
        a_sub = nx.adjacency_matrix(subgraph)
        r_sub = resistance_matrix(a_sub)
        m = len(subgraph)
        # add one to diagonal, invert, remove one from diagonal:
        c_sub = 1/(r_sub + np.eye(m)) - np.eye(m)
        subgraphC.append(c_sub)
    C = spla.block_diag(*subgraphC)
    # resort C so that it matches the original node list
    component_order = []
    for component in nx.connected_components(G):
        component_order += list(component)
    component_order = list(np.argsort(component_order))
    C = C[component_order,:]
    C = C[:,component_order]
    return C
