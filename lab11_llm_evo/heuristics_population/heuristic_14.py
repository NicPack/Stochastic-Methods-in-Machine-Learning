from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    This metaheuristic, inspired by the given algorithms, adaptively combines global exploration via differential evolution with local refinement using a Nelder-Mead simplex method, balancing the two based on budget and solution improvement.
    """

    dim = len(bounds)
    population_size = min(50, budget // 10)
    de_crossover_rate = 0.7
    de_mutation_factor = 0.5
    nelder_mead_iterations = 5

    # 1. Initialization (Differential Evolution)
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
        # 2.1 Differential Evolution Step
        for i in range(population_size):
            # Select three distinct individuals (a, b, c) from the population (excluding i)
            indices = np.random.choice(population_size, 3, replace=False)
            a, b, c = population[indices]

            # Create a trial vector (mutated vector)
            mutated_vector = a + de_mutation_factor * (b - c)

            # Apply crossover
            trial_vector = np.zeros(dim)
            for j in range(dim):
                if np.random.rand() < de_crossover_rate or j == np.random.randint(
                    dim
                ):  # Ensure at least one dimension is inherited from mutated vector.
                    trial_vector[j] = mutated_vector[j]
                else:
                    trial_vector[j] = population[i][j]

            # Clip to bounds
            trial_vector = np.clip(
                trial_vector, [b[0] for b in bounds], [b[1] for b in bounds]
            )

            # Evaluate trial vector
            trial_fitness = function(trial_vector)
            eval_count += 1

            # Selection: If trial vector is better than current individual, replace it
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial_vector.copy()

                # Update best solution
                if trial_fitness < best_val:
                    best_val = trial_fitness
                    best_x = trial_vector.copy()

        # Adaptive Nelder-Mead local search
        if (
            eval_count / budget > 0.5 and np.random.rand() < 0.2
        ):  # Start Local refinement with Nelder-Mead near the end
            from scipy.optimize import minimize

            def local_objective(x):
                return function(x)

            local_bounds = [(bounds[i][0], bounds[i][1]) for i in range(dim)]

            res = minimize(
                local_objective,
                best_x,
                method="Nelder-Mead",
                bounds=local_bounds,
                options={"maxiter": nelder_mead_iterations},
            )
            eval_count += res.nfev

            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
            if eval_count >= budget:
                break

    return best_val, best_x
