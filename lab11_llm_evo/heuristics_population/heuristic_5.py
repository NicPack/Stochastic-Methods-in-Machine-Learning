from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A novel metaheuristic algorithm for minimizing a black-box function with bound constraints.
    This implementation uses a combination of random search and local search with shrinking bounds.
    """

    dim = len(bounds)
    best_val = float("inf")
    best_x = None

    # Initial random search
    num_initial_samples = min(100, budget // 4)  # Adjust as needed
    for _ in range(num_initial_samples):
        x = np.array([np.random.uniform(low, high) for low, high in bounds])
        val = function(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()

    remaining_budget = budget - num_initial_samples

    # Iterative local search with shrinking bounds
    shrinkage_factor = 0.9  # Adjust as needed
    num_local_iterations = remaining_budget // 10  # Adjust as needed

    current_x = (
        best_x.copy()
        if best_x is not None
        else np.array([np.random.uniform(low, high) for low, high in bounds])
    )
    current_val = best_val if best_x is not None else function(current_x)

    for _ in range(num_local_iterations):
        # Define local bounds around the current best solution
        local_bounds = [
            (
                max(
                    bounds[i][0],
                    current_x[i] - shrinkage_factor * (bounds[i][1] - bounds[i][0]),
                ),
                min(
                    bounds[i][1],
                    current_x[i] + shrinkage_factor * (bounds[i][1] - bounds[i][0]),
                ),
            )
            for i in range(dim)
        ]

        # Sample a new point within the local bounds
        new_x = np.array([np.random.uniform(low, high) for low, high in local_bounds])

        # Evaluate the new point
        new_val = function(new_x)

        # Accept the new solution if it's better
        if new_val < current_val:
            current_val = new_val
            current_x = new_x.copy()

            # Update the global best if necessary
            if current_val < best_val:
                best_val = current_val
                best_x = current_x.copy()

    # Final evaluation to use the budget if possible
    if best_x is None:  # In the extremely unlikely case that it's never been set.
        x = np.array([np.random.uniform(low, high) for low, high in bounds])
        best_val = function(x)
        best_x = x

    return best_val, best_x
