"""
@file graph_metrics.py
@brief Implements functions for analyzing graph metrics such as density, 
       degree distribution, and clustering coefficients.
@author Sean Svihla
@details This file provides utility functions for calculating various metrics 
         for undirected graphs represented by their adjacency matrices. The 
         metrics include graph density, degree distribution, average degree, 
         clustering coefficients, and the average clustering coefficient.
"""


import numpy as np


def density(
    adj_matrix: np.ndarray
) -> float:
    """
    @brief Computes the density of an undirected graph.
    @param adj_matrix Adjacency matrix of the graph (assumed to be square and 
                      symmetric).
    @return Density of the graph, a float value in the range [0, 1].
    @details The density is defined as the ratio of the number of edges to the 
             maximum possible number of edges in a graph.
    """
    n_nodes = adj_matrix.shape[0]
    m_edges = np.sum(adj_matrix) / 2
    return 2 * m_edges / n_nodes / (n_nodes-1)


def degree_distribution(
    adj_matrix: np.ndarray
) -> np.ndarray:
    """
    @brief Computes the degree distribution of an undirected graph.
    @param adj_matrix Adjacency matrix of the graph (assumed to be square and 
                      symmetric).
    @return A 1D numpy array where each element represents the degree of the 
            corresponding node.
    @details The degree of a node is calculated as the sum of the entries in 
             the corresponding row of the adjacency matrix.
    """
    return np.sum(adj_matrix, axis=1)


def average_degree(
    adj_matrix: np.ndarray
) -> float:
    """
    @brief Computes the average degree of nodes in the graph.
    @param adj_matrix Adjacency matrix of the graph (assumed to be square and 
                      symmetric).
    @return Average degree of the nodes in the graph.
    @details The average degree is computed as the mean of the degree 
             distribution.
    """
    n_nodes = adj_matrix.shape[0]
    return np.sum(degree_distribution(adj_matrix)) / n_nodes


def clustering_coefficient(
    adj_matrix: np.ndarray
) -> np.ndarray:
    """
    @brief Computes the clustering coefficient for each node in the graph.
    @param adj_matrix Adjacency matrix of the graph (assumed to be square and 
                      symmetric).
    @return A 1D numpy array where each element represents the clustering 
            coefficient of the corresponding node.
    @details The clustering coefficient measures the fraction of triangles that 
             a node is part of, relative to the number of triangles it could 
             theoretically form.
    """
    triangles = adj_matrix @ adj_matrix @ adj_matrix / 2
    degs = np.sum(adj_matrix, axis=0)
    coefs = 2 * np.divide(np.diag(triangles), (degs * (degs - 1)))
    coefs[np.isnan(coefs) | np.isinf(coefs)] = 0
    return coefs


def average_clustering_coefficient(
    adj_matrix: np.ndarray
) -> np.ndarray:
    """
    @brief Computes the average clustering coefficient of the graph.
    @param adj_matrix Adjacency matrix of the graph (assumed to be square and 
                      symmetric).
    @return Average clustering coefficient of the graph.
    @details This is the mean value of the clustering coefficients of all 
             nodes in the graph.
    """
    n_nodes = adj_matrix.shape[0]
    return np.sum(clustering_coefficient(adj_matrix)) / n_nodes