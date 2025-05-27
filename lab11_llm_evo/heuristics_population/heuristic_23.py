from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    Inspired by population-based methods and adaptive search, this algorithm maintains a diverse population and refines search directions by learning from successful moves, dynamically adjusting step sizes and search biases based on population performance.
    """

    dim = len(bounds)
    pop_size = min(50, budget // 10)
    learning_rate = 0.1
    momentum = 0.5

    # 1. Initialization
    population = np.random.uniform(
        low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(pop_size, dim)
    )

    fitness = np.array([function(x) for x in population])
    eval_count = pop_size

    best_index = np.argmin(fitness)
    best_val = fitness[best_index]
    best_x = population[best_index].copy()

    # Initialize search directions (velocities)
    velocities = np.zeros_like(population)
    step_sizes = np.array(
        [(b[1] - b[0]) * 0.1 for b in bounds]
    )  # Initial step size for each dimension.

    # 2. Main Optimization Loop
    while eval_count < budget:
        for i in range(pop_size):
            # 1. Generate a new candidate solution based on learned direction

            # Update velocity based on past velocity and gradient toward the best
            gradient = best_x - population[i]
            velocities[i] = momentum * velocities[i] + (1 - momentum) * gradient

            new_x = (
                population[i] + learning_rate * velocities[i] * step_sizes
            )  # Apply step_sizes per dimension

            # 2. Boundary Handling
            new_x = np.clip(new_x, [b[0] for b in bounds], [b[1] for b in bounds])

            # 3. Evaluation
            new_fitness = function(new_x)
            eval_count += 1

            # 4. Update population and best solution
            if new_fitness < fitness[i]:
                fitness[i] = new_fitness
                population[i] = new_x.copy()

                if new_fitness < best_val:
                    best_val = new_fitness
                    best_x = new_x.copy()

            # 5. Adapt step sizes: Reduce step sizes if stuck or increase if improving
            if new_fitness > fitness[i]:  # No Improvement
                step_sizes *= 0.99  # Reduce step_size in all dimension
            else:
                # Increase step size in the dimension where we improved to explore further
                step_sizes += 0.01 * (
                    np.abs(new_x - population[i])
                )  # Increase step size where we moved

            step_sizes = np.clip(
                step_sizes, 0.0001, [(b[1] - b[0]) for b in bounds]
            )  # Ensure step sizes stay within reasonable bounds.

        if eval_count >= budget:
            break

    return best_val, best_x
