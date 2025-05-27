from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    This metaheuristic adaptively balances global exploration with local exploitation by using a clustering-based approach to identify promising regions,
    then applying a local search within these regions, adjusting the exploration-exploitation balance based on progress.
    """

    dimension = len(bounds)
    initial_population_size = min(100, budget // 10)
    local_search_iterations = 10

    # 1. Initial Population (Exploration)
    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(initial_population_size, dimension),
    )
    fitness = np.array([function(x) for x in population])
    evaluations = initial_population_size

    best_index = np.argmin(fitness)
    best_objective = fitness[best_index]
    best_solution = population[best_index].copy()

    # 2. Main Optimization Loop
    while evaluations < budget:
        # 3. Clustering (Identify Promising Regions) - Simplified K-Means
        num_clusters = min(
            5, initial_population_size // 5
        )  # Limit the number of clusters

        # Initialize centroids randomly from the population
        cluster_centers = population[
            np.random.choice(initial_population_size, size=num_clusters, replace=False)
        ]

        # Assign points to clusters
        clusters = [[] for _ in range(num_clusters)]
        cluster_indices = np.argmin(
            np.linalg.norm(
                population[:, None, :] - cluster_centers[None, :, :], axis=2
            ),
            axis=1,
        )

        for i, cluster_index in enumerate(cluster_indices):
            clusters[cluster_index].append(population[i])

        # Local Search on each cluster
        for i in range(num_clusters):
            if len(clusters[i]) > 0:
                # Find the best solution in the cluster
                cluster_solutions = np.array(clusters[i])
                cluster_fitnesses = np.array([function(x) for x in cluster_solutions])
                evaluations += len(clusters[i])

                cluster_best_index = np.argmin(cluster_fitnesses)
                cluster_best_solution = cluster_solutions[cluster_best_index].copy()

                # Local search on best cluster solutions
                current_x = cluster_best_solution.copy()
                current_val = cluster_fitnesses[cluster_best_index]
                step_size = 0.05  # Adaptive step size

                for _ in range(local_search_iterations):
                    # Perturb each dimension randomly
                    direction = np.random.uniform(-1, 1, size=dimension)
                    x_candidate = current_x + step_size * direction

                    # Clip to respect bounds
                    for j in range(dimension):
                        x_candidate[j] = np.clip(
                            x_candidate[j], bounds[j][0], bounds[j][1]
                        )

                    val_candidate = function(x_candidate)
                    evaluations += 1

                    if val_candidate < current_val:
                        current_val = val_candidate
                        current_x = x_candidate.copy()
                        step_size *= 1.1  # Increase stepsize for exploitation
                    else:
                        step_size *= 0.9  # Decrease stepsize for exploration

                    # Keep stepsize within sensible bounds
                    step_size = np.clip(step_size, 0.001, 0.1)

                    if current_val < best_objective:
                        best_objective = current_val
                        best_solution = current_x.copy()

                    if evaluations >= budget:
                        break
            if evaluations >= budget:
                break
        # Budget Check
        if evaluations >= budget:
            break

    return best_objective, best_solution
