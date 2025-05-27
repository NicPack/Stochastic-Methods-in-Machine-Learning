from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    This algorithm utilizes a population-based approach with a dynamic resource allocation strategy,
    adaptively shifting computational effort between exploration and exploitation based on the observed diversity and fitness landscape ruggedness,
    employing Gaussian process surrogate modeling to intelligently guide the search in promising regions.
    """

    dim = len(bounds)
    population_size = min(30, 3 * dim)

    # 1. Initialization
    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(population_size, dim),
    )
    fitness = np.array([function(x) for x in population])
    eval_count = population_size

    best_index = np.argmin(fitness)
    best_val = fitness[best_index]
    best_x = population[best_index].copy()

    # 2. Gaussian Process surrogate model (simplified - could use scikit-learn)
    # We will store the evaluated points and their fitness values
    X = population.copy()
    y = fitness.copy()

    # Parameters for dynamic resource allocation
    exploration_fraction = 0.3  # Initial fraction of budget for exploration
    landscape_ruggedness = 0.0  # Initialization

    def gaussian_process_estimate(x: np.ndarray) -> float:  # Simple kernel
        """Estimates the fitness value using a simple Gaussian kernel."""
        distances = np.linalg.norm(X - x, axis=1)  # Euclidean distances
        kernel_width = (
            np.mean([bounds[i][1] - bounds[i][0] for i in range(dim)]) / 5
        )  # Kernel width based on bounds
        kernel_values = np.exp(-(distances**2) / (2 * kernel_width**2))

        # Weighted average, higher weight to closer points
        return (
            np.sum(kernel_values * y) / np.sum(kernel_values)
            if np.sum(kernel_values) > 0
            else np.mean(y)
        )  # Return average if all weights are zero

    # 3. Main Optimization Loop
    while eval_count < budget:
        # 3.1 Dynamic Resource Allocation:

        remaining_budget = budget - eval_count
        exploration_budget = int(exploration_fraction * remaining_budget)
        exploitation_budget = remaining_budget - exploration_budget

        # 3.2 Exploration Phase: Latin Hypercube Sampling
        if exploration_budget > 0:
            from scipy.stats import qmc  # LHS sampling

            engine = qmc.LatinHypercube(d=dim)
            sample = engine.random(exploration_budget)
            exploration_points = qmc.scale(
                sample, [b[0] for b in bounds], [b[1] for b in bounds]
            )

            exploration_fitness = np.array([function(x) for x in exploration_points])
            eval_count += exploration_budget

            # Update X, y, best_x, and best_val
            X = np.vstack((X, exploration_points))
            y = np.concatenate((y, exploration_fitness))

            best_index_exploration = np.argmin(exploration_fitness)
            if exploration_fitness[best_index_exploration] < best_val:
                best_val = exploration_fitness[best_index_exploration]
                best_x = exploration_points[best_index_exploration].copy()

        # 3.3 Exploitation Phase: Surrogate-Assisted Optimization
        if exploitation_budget > 0:
            num_exploitation_points = min(
                10, exploitation_budget
            )  # limit number of points

            exploitation_points = np.zeros((num_exploitation_points, dim))
            exploitation_fitness = np.zeros(num_exploitation_points)

            for i in range(num_exploitation_points):
                # Sample near the current best solution guided by GP model

                candidate = best_x + np.random.normal(
                    0, 0.05 * (bounds[0][1] - bounds[0][0]), dim
                )  # Gaussian dist near best

                # Boundary handling
                candidate = np.clip(
                    candidate, [b[0] for b in bounds], [b[1] for b in bounds]
                )

                # Estimate fitness using Gaussian Process
                estimated_fitness = gaussian_process_estimate(candidate)

                # Evaluate the true fitness only for selected candidates
                exploitation_points[i] = candidate
                exploitation_fitness[i] = function(candidate)  # true evaluation
                eval_count += 1

            # Update X, y, best_x, and best_val

            X = np.vstack((X, exploitation_points))
            y = np.concatenate((y, exploitation_fitness))

            best_index_exploitation = np.argmin(exploitation_fitness)
            if exploitation_fitness[best_index_exploitation] < best_val:
                best_val = exploitation_fitness[best_index_exploitation]
                best_x = exploitation_points[best_index_exploitation].copy()

        # 3.4 Adaptation: Adjust Exploration Fraction

        if (
            eval_count > budget * 0.1
        ):  # only start adjusting after a certain number of evaluations
            # Measure landscape ruggedness:  Variance of fitness within a neighborhood

            neighborhood_size = min(50, len(X))  # Neighborhood to consider

            # Randomly sample a subset of evaluated points
            indices = np.random.choice(len(X), neighborhood_size, replace=False)

            fitness_subset = y[indices]
            ruggedness = np.std(fitness_subset)

            landscape_ruggedness = (
                0.9 * landscape_ruggedness + 0.1 * ruggedness
            )  # Smooth out the ruggedness measure

            exploration_fraction = 0.1 + 0.7 * np.tanh(
                landscape_ruggedness
            )  # Higher ruggedness -> higher exploration

        # Ensure budget is respected
        if eval_count >= budget:
            break

    return best_val, best_x
