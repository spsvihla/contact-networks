"""
@file sir_propagation.py
@brief Simulates the SIR model for disease propagation in a network.
@details This module contains a function that simulates the spread of infection 
         through a network represented by an adjacency matrix. The function runs 
         until either the infection dies out or a specified stop time is reached.
@author Sean Svihla
"""

from typing import Tuple
import numpy as np


def sir_propagation(
    adj_matrix: np.ndarray,
    infection_rate: float,
    recovery_rate: float,
    step_size: float,
    stop_time: float,
    initial_infected: int,
    initial_recovered: np.ndarray = np.array([], dtype=int)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    @brief Simulate the propagation of infection using the SIR model.
    
    @param adj_matrix A square adjacency matrix representing the graph of nodes,
                      where the value at (i, j) indicates the connection 
                      strength between nodes i and j.
    @param infection_rate The rate at which infection spreads between connected nodes.
    @param recovery_rate The rate at which infected nodes recover.
    @param step_size The time increment for each simulation step.
    @param stop_time The total duration for the simulation.
    @param initial_infected The index of the initial infected node.
    @param initial_recovered The initially recovered nodes (i.e., vaccinated).
    
    @return A tuple containing:
            - A set of susceptible nodes (those not infected).
            - A set of infectious nodes (those currently infected).
            - A set of recovered nodes (those who have recovered).
    """
    n_steps = np.inf if np.isinf(stop_time) else int(stop_time // step_size) 
    recovery_probability = recovery_rate * step_size

    n_nodes = adj_matrix.shape[0]
    susceptible = np.ones(n_nodes, dtype=bool) # all nodes start as susceptible
    infectious  = np.zeros(n_nodes, dtype=bool)
    recovered   = np.zeros(n_nodes, dtype=bool)

    if initial_infected in initial_recovered:
        raise ValueError('initial_infected cannot be in initial_recovered')

    susceptible[initial_infected] = False
    infectious[initial_infected]  = True
    recovered[initial_recovered]  = True

    step_count = 1
    while (step_count <= n_steps) and np.any(infectious):
        infectious_nodes = np.where(infectious)[0]

        # find susceptible neighbors
        neighbors = np.nonzero(adj_matrix[infectious_nodes, :])[1]
        susceptible_neighbors = neighbors[susceptible[neighbors]]

        # determine if neighbors are infected
        infection_probability = infection_rate * step_size
        infection_draws = np.random.uniform(size=susceptible_neighbors.size)
        infected_neighbors = np.unique(susceptible_neighbors[infection_draws < infection_probability])

        # update states
        infectious[infected_neighbors] = True
        susceptible[infected_neighbors] = False

        # determine if infectious nodes recover
        recovery_draws = np.random.uniform(size=infectious_nodes.size)
        recovered_nodes = infectious_nodes[recovery_draws < recovery_probability]

        # update states
        infectious[recovered_nodes] = False
        recovered[recovered_nodes] = True

        step_count += 1

        # if not np.any(susceptible):
        #     break

    return np.where(susceptible)[0], np.where(infectious)[0], np.where(recovered)[0]