"""
**Explanation:**

1.  **Common Idea:** The provided heuristics share the common idea of balancing exploration and exploitation within bound constraints to find the minimum of a black-box function. They use techniques like shrinking bounds, population-based search, random search with local refinement, and gradient-free methods to achieve this balance. Most of them use adaptive mechanisms to adjust the exploration-exploitation trade-off during the search process.

2.  **New Heuristic Description:** This metaheuristic adaptively balances global exploration with local refinement by using a population-based approach where individuals are attracted to both the global best and a dimension-specific, locally refined best, with dynamically adjusted step sizes for each dimension.

3.  **Implementation Details:**

    *   **Initialization:** Initializes a population within the given bounds. Each individual also maintains its own "local best" position. Dimension-specific step sizes are initialized, scaled to the bounds.
    *   **Perturbation:** Each individual's position is perturbed stochastically along each dimension, using the dimension-specific step sizes.
    *   **Local Best Update:** If the perturbed position yields a better fitness than the individual's current fitness, it's updated, and if better than individual's personal/local best it gets updated as well.
    *   **Step Size Adaptation:** If a move leads to improvement, the step size for that dimension is increased slightly; otherwise, it's decreased.
    *   **Global Attraction:** After each iteration, each individual is moved closer to the *global best* solution, as well as toward its own *local best* solution. This helps to exploit the best regions found so far. The combination of the global and local best is key in balancing exploration and exploitation.
    *   **Constraint Handling:** The new positions are clipped to respect the bound constraints.
    *   **Budget Management:** The algorithm terminates when the budget of function evaluations is exhausted."""

import numpy as np
from typing import Callable, List, Tuple

def new_metaheuristic(
	function: Callable[[np.ndarray], float], 
    bounds: List[Tuple[float, float]], 
    budget: int
) -> Tuple[float, np.ndarray]:
    """
    This metaheuristic adaptively balances global exploration with local refinement by using a population-based approach where individuals are attracted to both the global best and a dimension-specific, locally refined best, with dynamically adjusted step sizes for each dimension.
    """
    dim = len(bounds)
    pop_size = min(50, budget // 10)

    # 1. Initialization
    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(pop_size, dim)
    )

    fitness = np.array([function(x) for x in population])
    eval_count = pop_size

    best_index = np.argmin(fitness)
    best_val = fitness[best_index]
    best_x = population[best_index].copy()

    # Initialize dimension-specific bests and step sizes
    local_best_x = population.copy()
    local_best_fitness = fitness.copy()
    step_sizes = np.array([(b[1] - b[0]) * 0.1 for b in bounds])  # Initial step sizes, scaled to bounds

    # 2. Main Optimization Loop
    while eval_count < budget:
        for i in range(pop_size):
            # Perturb each dimension with individual step sizes
            direction = np.random.normal(0, 1, size=dim)
            new_x = population[i] + step_sizes * direction

            # Clip to respect bounds
            new_x = np.clip(new_x, [b[0] for b in bounds], [b[1] for b in bounds])
            
            new_val = function(new_x)
            eval_count += 1

            # Update individual position if better
            if new_val < fitness[i]:
                fitness[i] = new_val
                population[i] = new_x.copy()
                
                #Update local best for individual i, if better
                if new_val < local_best_fitness[i]:
                    local_best_fitness[i] = new_val
                    local_best_x[i] = new_x.copy()

                # Step size adaptation (simplified): if a move leads to improvement, increase step size slightly, otherwise decrease
                step_sizes = np.clip(step_sizes * (1 + 0.01), 0.0001, [b[1] - b[0] for b in bounds])

            else:
                #Step size reduction
                step_sizes = np.clip(step_sizes * (1 - 0.01), 0.0001, [b[1] - b[0] for b in bounds])

            # Update global best
            if new_val < best_val:
                best_val = new_val
                best_x = new_x.copy()


        # Global attraction: Move each individual closer to the global best, using its local best as an intermediary
        for i in range(pop_size):
            #Combination of global and local best
            attraction_vector = 0.5 * (best_x - population[i]) + 0.5*(local_best_x[i]-population[i])
            population[i] = np.clip(population[i] + 0.1 * attraction_vector, [b[0] for b in bounds], [b[1] for b in bounds]) #Move towards the combination
        

        if eval_count >= budget:
            break

    return best_val, best_x