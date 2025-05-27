from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    Inspired by trust-region methods, differential evolution, and adaptive exploration, this algorithm focuses on iteratively refining promising solutions by creating and evaluating variations within dynamically adjusted, dimension-specific bounds, then biasing the search towards the best-performing regions.
    """

    dim = len(bounds)
    pop_size = min(50, budget // 10)
    exploration_rate = 0.8  # Probability of exploration

    # 1. Initialization
    population = np.random.uniform(
        low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim)
    )

    fitness = np.array([function(x) for x in population])
    eval_count = pop_size

    best_index = np.argmin(fitness)
    best_val = fitness[best_index]
    best_x = population[best_index].copy()

    adaptive_bounds = [list(b) for b in bounds]  # Start with original bounds

    # 2. Main Optimization Loop
    while eval_count < budget:
        new_population = []
        for i in range(pop_size):
            if np.random.rand() < exploration_rate:
                # Exploration - Sample from full bounds with shrinking exploration rate
                new_x = np.random.uniform(
                    low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=dim
                )
            else:
                # Exploitation - Generate variations near the current best solution,
                # using adaptive bounds.

                # Differential evolution with adaptive bound shrink
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + 0.5 * (population[b] - population[c])

                # Crossover with best solution
                trial = np.zeros(dim)
                for j in range(dim):
                    if np.random.rand() < 0.7 or j == np.random.randint(dim):
                        trial[j] = mutant[j]
                    else:
                        trial[j] = best_x[j]

                # Repair by Clipping to adaptive bounds
                new_x = np.clip(
                    trial,
                    [b[0] for b in adaptive_bounds],
                    [b[1] for b in adaptive_bounds],
                )

            new_population.append(new_x)

        new_population = np.array(new_population)
        new_fitness = np.array([function(x) for x in new_population])
        eval_count += pop_size

        # Combine old and new populations and select the best.
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))

        sorted_indices = np.argsort(combined_fitness)[:pop_size]
        population = combined_population[sorted_indices]
        fitness = combined_fitness[sorted_indices]

        best_index = np.argmin(fitness)
        if fitness[best_index] < best_val:
            best_val = fitness[best_index]
            best_x = population[best_index].copy()

        # Adaptive Bounds
        for j in range(dim):
            range_j = bounds[j][1] - bounds[j][0]
            width = 0.1 * range_j  # Width of Adaptive Bound

            adaptive_bounds[j][0] = max(bounds[j][0], best_x[j] - width)
            adaptive_bounds[j][1] = min(bounds[j][1], best_x[j] + width)

        # Decrease Exploration Rate
        exploration_rate *= 0.99
        exploration_rate = np.clip(exploration_rate, 0.1, 0.8)

        if eval_count >= budget:
            break

    return best_val, best_x
