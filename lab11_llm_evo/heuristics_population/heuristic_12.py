from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A metaheuristic algorithm that combines the exploration of a diverse population with the intensification of a trust-region-like method, adapting the trust region size based on search progress.
    """

    dim = len(bounds)
    population_size = min(50, budget // 10)
    trust_region_size = 0.2  # Initial trust region size as a fraction of the bounds

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

    # 2. Main Optimization Loop
    while eval_count < budget:
        # 2.1 Selection: Select the best individual as the center of the trust region.

        # 2.2 Trust Region Exploration: Generate new candidate solutions within the trust region.
        new_population = []
        for _ in range(population_size):
            # Define local bounds for trust region
            local_bounds = [
                (
                    max(
                        bounds[i][0],
                        best_x[i] - trust_region_size * (bounds[i][1] - bounds[i][0]),
                    ),
                    min(
                        bounds[i][1],
                        best_x[i] + trust_region_size * (bounds[i][1] - bounds[i][0]),
                    ),
                )
                for i in range(dim)
            ]

            # Sample within local bounds
            new_x = np.array(
                [np.random.uniform(low, high) for low, high in local_bounds]
            )
            new_population.append(new_x)

        new_population = np.array(new_population)
        new_fitness = np.array([function(x) for x in new_population])
        eval_count += population_size

        # 2.3 Update Population: Combine old and new populations and select the best.
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))

        sorted_indices = np.argsort(combined_fitness)[
            :population_size
        ]  # Select top population_size individuals
        population = combined_population[sorted_indices]
        fitness = combined_fitness[sorted_indices]

        # 2.4 Update Best Solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < best_val:
            best_val = fitness[best_index]
            best_x = population[best_index].copy()

        # 2.5 Adjust Trust Region Size: Adapt the trust region size based on the progress.
        # Shrink trust region if the best solution hasn't changed much recently, otherwise expand.

        if eval_count / budget > 0.2:
            trust_region_size *= 0.95  # Shrink trust region as budget used up

        trust_region_size = np.clip(trust_region_size, 0.01, 0.5)

        if eval_count >= budget:
            break

    return best_val, best_x
