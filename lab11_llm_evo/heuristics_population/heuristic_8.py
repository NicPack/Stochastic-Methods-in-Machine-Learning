from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A metaheuristic that combines aspects of particle swarm optimization (PSO) and covariance matrix adaptation evolution strategy (CMA-ES) with a budget-aware adaptation mechanism.
    """
    dim = len(bounds)
    population_size = min(40, 4 * dim)  # dynamic population size

    # Initialization
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

    # PSO parameters
    inertia = 0.7
    cognitive_coeff = 1.5
    social_coeff = 1.5
    velocities = np.zeros_like(population)

    # CMA-ES parameters
    mean = np.mean(population, axis=0)
    covariance = (
        np.eye(dim) * 0.1 * np.max([b[1] - b[0] for b in bounds])
    )  # initial covariance

    c_sigma = 0.3  # Learning rate for step size adaptation
    damps = 1 + (dim / 3)  # Damping parameter for step-size

    p_sigma = np.zeros(dim)  # Evolution Path for step size adaptation
    mu_eff = population_size / 4.0  # Variance effectiveness

    while eval_count < budget:
        # Budget aware adaptation

        time_ratio = eval_count / budget  # Fraction of budget spent
        inertia = (
            0.7 - 0.4 * time_ratio
        )  # Linearly decrease inertia (exploration -> exploitation)

        for i in range(population_size):
            # PSO update

            # Update velocity
            r1, r2 = np.random.rand(2)
            velocities[i] = (
                inertia * velocities[i]
                + cognitive_coeff * r1 * (best_x - population[i])
                + social_coeff * r2 * (population[np.argmin(fitness)] - population[i])
            )

            # Apply velocity
            new_x = population[i] + velocities[i]

            # Clip to bounds
            new_x = np.clip(new_x, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate candidate
            new_val = function(new_x)
            eval_count += 1

            if new_val < fitness[i]:
                fitness[i] = new_val
                population[i] = new_x.copy()

                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()

        # Adapt mean position
        mean = np.mean(population, axis=0)

        # Adaptive Covariance Matrix
        z = np.random.multivariate_normal(np.zeros(dim), covariance, population_size)
        population = mean + z

        # Boundary Clipping
        population = np.clip(population, [b[0] for b in bounds], [b[1] for b in bounds])

        # Evaluate new population
        fitness = np.array([function(x) for x in population])
        eval_count += population_size

        if eval_count > budget:
            break

        best_index = np.argmin(fitness)
        if fitness[best_index] < best_val:
            best_val = fitness[best_index]
            best_x = population[best_index].copy()

    return best_val, best_x
