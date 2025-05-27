"""Let's analyze the provided heuristics and design a new one.

**1. Common Idea:**

The common thread among these heuristics is the iterative improvement of a solution (or a population of solutions) within the defined bounds, using a combination of exploration (random search, mutation) and exploitation (local search, selection) strategies, all while respecting the budget constraint. Many use a population-based approach to encourage diversity.

**2. New Heuristic Description:**

The new heuristic will be a population-based algorithm that adapts its search behavior by probabilistically switching between global exploration and local exploitation phases, guided by the performance of the population.

**3. Implementation:**
"""

from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A metaheuristic algorithm that dynamically balances exploration and exploitation
    based on population performance.
    """
    dim = len(bounds)
    population_size = min(50, 5 * dim)

    # Initialize population
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

    # Parameters
    exploration_prob = 0.5  # Initial exploration probability
    exploitation_intensity = 0.1  # Initial exploitation intensity

    while eval_count < budget:
        for i in range(population_size):
            if np.random.rand() < exploration_prob:
                # Exploration: Randomly perturb the solution
                new_x = np.random.uniform(
                    low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=dim
                )
            else:
                # Exploitation: Move towards a randomly selected better solution

                other_index = np.random.randint(population_size)
                while other_index == i:
                    other_index = np.random.randint(population_size)

                if fitness[other_index] < fitness[i]:
                    direction = population[other_index] - population[i]
                else:
                    direction = (
                        population[i] - population[other_index]
                    )  # Or move *away* if worse... helps sometimes

                new_x = population[i] + exploitation_intensity * direction

                # Boundary handling
                new_x = np.clip(new_x, [b[0] for b in bounds], [b[1] for b in bounds])

            new_val = function(new_x)
            eval_count += 1

            if new_val < fitness[i]:
                fitness[i] = new_val
                population[i] = new_x.copy()

                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()

        # Adaptation: Adjust exploration probability based on population variance
        # High variance -> more exploration, low variance -> more exploitation

        std_devs = np.std(population, axis=0)
        average_std = np.mean(std_devs)

        # Adapt the exploration prob based on the population variance and time
        exploration_prob = 0.2 + 0.8 * (
            1 - (eval_count / budget)
        )  # linear decay of exploration
        exploration_prob += 0.3 * np.tanh(
            average_std
        )  # add some non-linear exploration

        # Clamp exploration_prob
        exploration_prob = np.clip(exploration_prob, 0.1, 0.9)

        # Adapt exploitation intensity based on success.
        if best_val < np.min(fitness):
            exploitation_intensity *= 1.05  # Increased the exploit rate slightly
        else:
            exploitation_intensity *= 0.95  # Decreased it.

        exploitation_intensity = np.clip(exploitation_intensity, 0.01, 0.2)

        if eval_count >= budget:
            break

    return best_val, best_x
