"""
********
Features
********

Calculation of features for NetSimile algorithm.
"""
import networkx as nx
import numpy as np
from scipy import stats

from netcomp.linalg import _eps

def get_features(A):
    """Feature grabber for NetSimile algorithm. Features used are

        1. Degree of node
        2. Clustering coefficient of node
        3. Average degree of node's neighbors
        4. Average clustering coefficient of node's neighbors
        5. Number of edges in node's egonet
        6. Number of neighbors of node's egonet
        7. Number of outgoing edges from node's egonet

    Parameters
    ---------
    A : NumPy matrix
        Adjacency matrix of graph in question. Preferably a SciPy sparse matrix
        for large graphs.

    Returns
    -------
    feature_mat : NumPy array
        An n by 7 array of features, where n = A.shape[0]

    References
    -----
    [Berlingerio 2012]

    """
    try:
        G = nx.from_scipy_sparse_matrix(A)
    except AttributeError:
        G = nx.from_numpy_matrix(A)
    n = len(G)
    # degrees, array so we can slice nice
    d_vec = np.array(list(G.degree().values()))
    # list of clustering coefficient
    clust_vec = np.array(list(nx.clustering(G).values()))
    neighbors = [G.neighbors(i) for i in range(n)]
    # average degree of neighbors (0 if node is isolated)
    neighbor_deg = [d_vec[neighbors[i]].sum()/d_vec[i]
                    if d_vec[i]>_eps else 0 for i in range(n)]
    # avg. clustering coefficient of neighbors (0 if node is isolated)
    neighbor_clust = [clust_vec[neighbors[i]].sum()/d_vec[i] 
                    if d_vec[i]>_eps else 0 for i in range(n)]
    egonets = [nx.ego_graph(G,i) for i in range(n)]
    # number of edges in egonet
    ego_size = [G.number_of_edges() for G in egonets]
    # number of neighbors of egonet
    ego_neighbors = [len(set.union(*[set(neighbors[j])
                                     for j in egonets[i].nodes()]) -
                         set(egonets[i].nodes()))
                     for i in range(n)]
    # number of edges outgoing from egonet
    outgoing_edges = [len([edge for edge in G.edges(egonets[i].nodes()) 
                           if edge[1] not in egonets[i].nodes()]) 
                      for i in range(n)]
    # use mat.T so that each node is a row (standard format)
    feature_mat = np.array([d_vec,clust_vec,neighbor_deg,neighbor_clust,
                            ego_size,ego_neighbors,outgoing_edges]).T 
    return feature_mat




def aggregate_features(feature_mat,row_var=False,as_matrix=False):
    """Returns column-wise descriptive statistics of a feature matrix.

    Parameters
    ----------
    feature_mat : NumPy array
        Matrix on which statistics are to be calculated. Assumed to be formatted
        so each row is an observation (a node, in the case of NetSimile).

    row_var : Boolean, optional (default=False)
        If true, then each variable has it's own row, and statistics are
        computed along rows rather than columns.

    as_matrix : Boolean, optional (default=False)
        If true, then description is returned as matrix. Otherwise, it is
        flattened into a vector.

    Returns
    -------
    description : NumPy array
        Descriptive statistics of feature_mat

    Notes
    -----

    References
    ----------
    """
    axis = int(row_var) # 0 if column-oriented, 1 if not
    description = np.array([feature_mat.mean(axis=axis),
                            np.median(feature_mat,axis=axis),
                            np.std(feature_mat,axis=axis),
                            stats.skew(feature_mat,axis=axis),
                            stats.kurtosis(feature_mat,axis=axis)])
    if not as_matrix:
        description = description.flatten()
    return description
