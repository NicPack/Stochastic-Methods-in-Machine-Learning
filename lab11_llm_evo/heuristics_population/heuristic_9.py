from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A metaheuristic combining a simplified Simulated Annealing (SA) with periodic restarts and adaptive temperature scaling based on the budget and search progress.
    """

    dimension = len(bounds)

    # Initialization
    current_solution = np.random.uniform(
        low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=dimension
    )
    current_energy = function(current_solution)
    evaluations = 1

    best_solution = current_solution.copy()
    best_energy = current_energy

    # Initial temperature
    temperature = (
        np.max([b[1] - b[0] for b in bounds]) / 10
    )  # Based on the range of the search space

    # Main loop
    while evaluations < budget:
        # Generate a neighbor (slight perturbation)
        neighbor = current_solution.copy()
        for i in range(dimension):
            neighbor[i] += np.random.normal(
                0, temperature / 10
            )  # Perturbation size decreases with temperature
            neighbor[i] = np.clip(
                neighbor[i], bounds[i][0], bounds[i][1]
            )  # Clip to bounds

        # Evaluate neighbor
        neighbor_energy = function(neighbor)
        evaluations += 1

        # Acceptance probability (Metropolis criterion)
        delta_energy = neighbor_energy - current_energy
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            current_solution = neighbor.copy()
            current_energy = neighbor_energy

            # Update best solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_solution = current_solution.copy()

        # Temperature update (cooling schedule)
        # Adaptive temperature schedule: reduce temperature more aggressively early on
        # and then fine-tune later.
        time_ratio = evaluations / budget
        if time_ratio < 0.5:
            temperature *= 0.95  # Faster cooling initially
        else:
            temperature *= 0.99  # Slower cooling as we get closer to the budget

        # Restart mechanism (periodic) to escape local optima
        if (
            evaluations % (budget // 10) == 0 and evaluations < budget
        ):  # restart every 10% of budget
            current_solution = np.random.uniform(
                low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=dimension
            )
            current_energy = function(current_solution)
            evaluations += 1
            temperature = (
                np.max([b[1] - b[0] for b in bounds]) / 10
            )  # reset temperature

    return best_energy, best_solution
