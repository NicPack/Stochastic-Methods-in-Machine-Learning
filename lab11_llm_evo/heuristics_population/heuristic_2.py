from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A novel metaheuristic algorithm for minimizing a black-box function with bound constraints.

    This algorithm uses a combination of random search and local search,
    with a focus on exploration in the early stages and exploitation in the later stages.

    Args:
        function: The objective function to minimize.
        bounds: A list of (lower, upper) pairs delimiting the search space for each dimension.
        budget: The total number of objective-function evaluations the algorithm may perform.

    Returns:
        A tuple containing the best objective value found and the corresponding decision vector.
    """

    dim = len(bounds)
    best_val = float("inf")
    best_x = None
    eval_count = 0

    # Initial random search phase (exploration)
    num_initial_samples = min(
        100, budget // 4
    )  # Evaluate at least 100 or 25% of budget
    for _ in range(num_initial_samples):
        x = np.array([np.random.uniform(low, high) for low, high in bounds])
        val = function(x)
        eval_count += 1

        if val < best_val:
            best_val = val
            best_x = x.copy()

    # Local search phase (exploitation)
    while eval_count < budget:
        # Generate a candidate solution near the best solution
        mutation_scale = (
            budget - eval_count
        ) / budget  # Gradually reduce the mutation scale

        x_candidate = best_x + np.random.normal(0, mutation_scale, size=dim)

        # Clip the candidate to the bounds
        for i in range(dim):
            x_candidate[i] = np.clip(x_candidate[i], bounds[i][0], bounds[i][1])

        val_candidate = function(x_candidate)
        eval_count += 1

        if val_candidate < best_val:
            best_val = val_candidate
            best_x = x_candidate.copy()

        # Random restart if stuck, but only occasionally
        if np.random.rand() < 0.01 and eval_count < budget:
            x = np.array([np.random.uniform(low, high) for low, high in bounds])
            val = function(x)
            eval_count += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()

    return best_val, best_x
