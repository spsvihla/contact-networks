"""
@file   sampling_methods.py
@brief  Provides sampling methods for graphs represented as adjacency matrices.
@author Sean Svila
"""


import numpy as np
from . import graph_metrics


def node_sampling(
    adj_matrix: np.ndarray, 
    n_nodes: int
) -> np.ndarray:
    """
    @brief Samples a subgraph by selecting `n_nodes` from the adjacency matrix.
    @param adj_matrix The input adjacency matrix representing the graph.
    @param n_nodes The number of nodes to sample.
    @return A subgraph as a new adjacency matrix containing only the sampled 
            nodes.
    """
    sub_matrix = np.zeros(adj_matrix.shape)
    vs = np.random.choice(adj_matrix.shape[0], size=n_nodes, replace=False)
    ix = np.ix_(vs, vs)
    sub_matrix[ix] = adj_matrix[ix]
    return sub_matrix


def edge_sampling(
    adj_matrix: np.ndarray, 
    n_edges: int
) -> np.ndarray:
    """
    @brief Samples a subgraph by selecting `n_edges` from the adjacency matrix.
    @param adj_matrix The input adjacency matrix representing the graph.
    @param n_edges The number of edges to sample.
    @return A subgraph as a new adjacency matrix containing only the sampled 
            edges.
    """
    sub_matrix = np.zeros(adj_matrix.shape)
    rows, cols = np.nonzero(adj_matrix)
    edge_idxs = np.array((rows, cols))[:, rows <= cols].T
    edges = np.random.choice(edge_idxs.shape[0], size=n_edges, replace=False)
    for edge in edges:
        row, col = edge_idxs[edge]
        sub_matrix[row, col] = 1
        sub_matrix[col, row] = 1
    return sub_matrix


def metropolis_hastings_rw(
    adj_matrix: np.ndarray,
    n_edges: int
) -> np.ndarray:
    """
    @brief Performs a Metropolis-Hastings random walk on the adjacency matrix 
           to sample edges.
    @param adj_matrix The input adjacency matrix representing the graph.
    @param n_edges The number of edges to sample using the random walk.
    @return A subgraph as a new adjacency matrix containing the sampled edges 
            from the random walk.
    """
    sub_matrix = np.zeros(adj_matrix.shape)
    degrees = graph_metrics.degree_distribution(adj_matrix)
    current_node = np.random.choice(adj_matrix.shape[0])
    edges_added = 0
    while edges_added < n_edges:
        neighbors = np.nonzero(adj_matrix[current_node, :])[0]
        candidate = np.random.choice(neighbors)
        acceptance_ratio = degrees[current_node] / degrees[candidate]
        if np.random.uniform(0, 1) <= acceptance_ratio:
            sub_matrix[current_node, candidate] = 1
            sub_matrix[candidate, current_node] = 1
            current_node = candidate
            edges_added += 1
    return sub_matrix


def frontier_sampling(
    adj_matrix: np.ndarray,
    n_edges: int,
    n_nodes: int = None
) -> np.ndarray:
    """
    @brief Performs frontier-based sampling to select edges and nodes from the 
           adjacency matrix.
    @param adj_matrix The input adjacency matrix representing the graph.
    @param n_edges The number of edges to sample.
    @param n_nodes The number of nodes to sample initially for the frontier.
    @return A subgraph as a new adjacency matrix containing the sampled edges 
            based on frontier nodes.
    """
    size = adj_matrix.shape[0]
    if not n_nodes:
        n_nodes = size // 5
    sub_matrix = np.zeros(adj_matrix.shape)
    degrees = graph_metrics.degree_distribution(adj_matrix)
    frontier = np.random.choice(size, size=n_nodes, replace=False)
    edges_added = 0
    while edges_added < n_edges:
        density = degrees[frontier] / np.sum(degrees[frontier])
        current_node = row = np.random.choice(frontier, p=density)
        neighbors = np.nonzero(adj_matrix[current_node, :])[0]
        new_node = col = np.random.choice(neighbors)
        frontier[np.where(frontier == current_node)[0][0]] = new_node
        sub_matrix[row, col] = 1
        sub_matrix[col, row] = 1
        edges_added += 1
    return sub_matrix


def snowball_expansion_sampling(
    adj_matrix: np.ndarray,
    n_nodes: int
) -> np.ndarray:
    """
    @brief Perform snowball expansion sampling on a graph represented by an 
           adjacency matrix.

    The function starts with a random node and iteratively selects new nodes 
    from the neighborhood of the current set of nodes, maximizing the size of 
    the neighborhood difference (i.e., choosing the node that connects to the 
    most nodes outside the current set of selected nodes).

    @param adj_matrix The adjacency matrix representing the graph.
    @param n_nodes The number of nodes to sample.
    @return A submatrix containing the sampled nodes and their edges from the 
            original graph.
    """
    size = adj_matrix.shape[0]
    sub_matrix = np.zeros(adj_matrix.shape)
    adj_list = [set(np.nonzero(adj_matrix[i])[0]) for i in range(size)]
    nodes = {np.random.choice(size)}
    suns = set.union(*(adj_list[v] for v in nodes)) # S union N(S)
    for _ in range(n_nodes - 1):
        neighbors = suns.difference(nodes)
        new_node = max(neighbors, key=lambda n: len(adj_list[n].difference(suns)))
        nodes.add(new_node)
        suns.update(adj_list[new_node])
    nodes = list(nodes)
    ix = np.ix_(nodes, nodes)
    sub_matrix[ix] = adj_matrix[ix]
    return sub_matrix