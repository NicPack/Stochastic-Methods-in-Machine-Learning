from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    This metaheuristic employs a population-based approach with a dynamic resource allocation scheme that adaptively shifts computational effort between global exploration and local refinement phases, guided by the observed diversity and fitness improvements within the population.
    """
    dim = len(bounds)
    pop_size = min(50, budget // 10)
    exploration_probability = 0.5

    # Initialization
    population = np.random.uniform(
        low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim)
    )
    fitness = np.array([function(x) for x in population])
    eval_count = pop_size

    best_index = np.argmin(fitness)
    best_val = fitness[best_index]
    best_x = population[best_index].copy()

    while eval_count < budget:
        new_population = np.zeros_like(population)
        new_fitness = np.zeros_like(fitness)

        for i in range(pop_size):
            if np.random.rand() < exploration_probability:
                # Exploration: Generate a completely new solution
                new_x = np.random.uniform(
                    low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=dim
                )
            else:
                # Refinement: Perturb the current best solution
                mutation_scale = 0.1 * (
                    np.array([b[1] - b[0] for b in bounds])
                )  # Dimension-specific
                new_x = best_x + np.random.normal(0, mutation_scale)

                # Clip to bounds
                new_x = np.clip(new_x, [b[0] for b in bounds], [b[1] for b in bounds])

            new_fitness[i] = function(new_x)
            eval_count += 1
            new_population[i] = new_x

            if new_fitness[i] < best_val:
                best_val = new_fitness[i]
                best_x = new_x.copy()

        # Update population based on fitness
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))

        sorted_indices = np.argsort(combined_fitness)[:pop_size]
        population = combined_population[sorted_indices]
        fitness = combined_fitness[sorted_indices]

        # Dynamically adjust exploration probability
        fitness_std = np.std(fitness)
        if fitness_std < 1e-6:
            exploration_probability = min(exploration_probability * 1.1, 0.9)
        else:
            exploration_probability = max(exploration_probability * 0.9, 0.1)

        if eval_count >= budget:
            break

    return best_val, best_x
