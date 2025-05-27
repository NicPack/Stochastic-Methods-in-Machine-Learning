from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    Implements a simplified version of Differential Evolution (DE) for optimization.

    Args:
        function: The objective function to minimize.
        bounds: A list of (lower, upper) bounds for each dimension.
        budget: The total number of function evaluations allowed.

    Returns:
        A tuple containing the best objective value found and the corresponding decision vector.
    """

    dim = len(bounds)
    pop_size = min(10 * dim, 50)  # Population size, capped at 50
    mutation_factor = 0.5
    crossover_rate = 0.7

    # Initialize population within bounds
    population = np.zeros((pop_size, dim))
    for i in range(dim):
        population[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], pop_size)

    # Evaluate initial population
    fitness = np.array([function(x) for x in population])
    eval_count = pop_size

    best_index = np.argmin(fitness)
    best_fitness = fitness[best_index]
    best_solution = population[best_index].copy()

    # Optimization loop
    while eval_count < budget:
        for i in range(pop_size):
            # Mutation
            indices = list(range(pop_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = population[a] + mutation_factor * (population[b] - population[c])

            # Crossover
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.rand() < crossover_rate or j == np.random.randint(dim):
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i][j]

            # Repair (clip)
            for j in range(dim):
                trial[j] = np.clip(trial[j], bounds[j][0], bounds[j][1])

            # Evaluation
            trial_fitness = function(trial)
            eval_count += 1

            # Selection
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial

                # Update best solution
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial.copy()

            if eval_count >= budget:
                break

    return best_fitness, best_solution
