from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    This metaheuristic uses a self-organizing map (SOM) to map solutions to a grid, then explores and exploits based on the fitness landscape represented by the SOM, dynamically adjusting learning rates and neighborhood sizes for exploration-exploitation balance.
    """

    dimension = len(bounds)
    grid_size = int(
        np.sqrt(min(budget // 10, 100))
    )  # Size of the SOM grid (e.g., 10x10)
    learning_rate = 0.5
    neighborhood_radius = grid_size // 2
    learning_rate_decay = 0.99
    neighborhood_decay = 0.95

    # 1. Initialize SOM
    som = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(grid_size, grid_size, dimension),
    )

    # 2. Initialize Fitness Map
    fitness_map = np.full((grid_size, grid_size), float("inf"))

    # 3. Initialize Best Solution
    best_solution = None
    best_objective = float("inf")
    evaluations = 0

    def evaluate(solution):
        nonlocal evaluations
        evaluations += 1
        return function(solution)

    # Main loop
    while evaluations < budget:
        # 1. Sample a Random Solution (Exploration)
        solution = np.random.uniform(
            low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=dimension
        )

        objective = evaluate(solution)

        # Update best solution
        if objective < best_objective:
            best_objective = objective
            best_solution = solution.copy()

        # Find the Best Matching Unit (BMU) in the SOM
        distances = np.linalg.norm(som - solution, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        bmu_row, bmu_col = bmu_index

        # Update fitness map
        if objective < fitness_map[bmu_row, bmu_col]:
            fitness_map[bmu_row, bmu_col] = objective
            som[bmu_row, bmu_col] = solution.copy()

        # Update SOM neighborhood
        for i in range(
            max(0, bmu_row - neighborhood_radius),
            min(grid_size, bmu_row + neighborhood_radius + 1),
        ):
            for j in range(
                max(0, bmu_col - neighborhood_radius),
                min(grid_size, bmu_col + neighborhood_radius + 1),
            ):
                dist_to_bmu = np.sqrt((i - bmu_row) ** 2 + (j - bmu_col) ** 2)
                if dist_to_bmu <= neighborhood_radius:
                    som[i, j] += learning_rate * (solution - som[i, j])

        # Adjust exploration-exploitation balance
        learning_rate *= learning_rate_decay
        neighborhood_radius *= neighborhood_decay
        neighborhood_radius = max(1, int(neighborhood_radius))

        if evaluations >= budget:
            break

    return best_objective, best_solution
