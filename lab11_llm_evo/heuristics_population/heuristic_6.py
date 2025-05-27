from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A new metaheuristic algorithm that combines a diversified initial sampling with a constraint-respecting gradient-free local search, adaptively balancing exploration and exploitation.
    """

    dim = len(bounds)
    best_val = float("inf")
    best_x = None
    eval_count = 0

    # 1. Diversified Initial Sampling (Latin Hypercube Sampling)
    num_initial_samples = min(10 * dim, budget // 5)  # Exploration phase

    # Latin Hypercube Sampling
    samples = np.zeros((num_initial_samples, dim))
    for i in range(dim):
        intervals = np.linspace(bounds[i][0], bounds[i][1], num_initial_samples + 1)
        lower_bounds = intervals[:-1]
        upper_bounds = intervals[1:]
        samples[:, i] = np.random.uniform(lower_bounds, upper_bounds)
        np.random.shuffle(samples[:, i])  # Shuffle within each dimension

    for x in samples:
        val = function(x)
        eval_count += 1

        if val < best_val:
            best_val = val
            best_x = x.copy()

    # 2. Constraint-Respecting Gradient-Free Local Search
    remaining_budget = budget - eval_count
    if remaining_budget <= 0:
        return best_val, best_x

    local_search_iterations = remaining_budget
    current_x = best_x.copy()
    current_val = best_val

    step_size = 0.1  # Initial step size
    success_rate = 0.5  # Probability of success to adjust step size

    for _ in range(local_search_iterations):
        # Perturb each dimension randomly
        direction = np.random.uniform(-1, 1, size=dim)
        x_candidate = current_x + step_size * direction

        # Clip to respect bounds
        for i in range(dim):
            x_candidate[i] = np.clip(x_candidate[i], bounds[i][0], bounds[i][1])

        val_candidate = function(x_candidate)
        eval_count += 1

        if val_candidate < current_val:
            current_val = val_candidate
            current_x = x_candidate.copy()

            if val_candidate < best_val:
                best_val = val_candidate
                best_x = x_candidate.copy()

            success_rate = 0.8  # Increase the probability if successful
            step_size *= 1 + np.random.normal(
                0, 0.01
            )  # Tiny random increase if successful
        else:
            success_rate = 0.2  # Decrease probability if unsuccessful
            step_size *= 1 - np.random.normal(
                0, 0.01
            )  # Tiny random decrease if unsuccessful

        step_size = np.clip(step_size, 0.0001, 0.1)  # Clamp the stepsize

        if eval_count >= budget:
            break

    return best_val, best_x
