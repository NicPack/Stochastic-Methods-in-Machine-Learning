from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A population-based metaheuristic that simulates the diffusion and aggregation of particles within the search space, where particle density guides the search towards promising regions and dynamic repulsion prevents premature convergence.
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
    diffusion_rate = 0.1  # Controls the magnitude of diffusion
    aggregation_strength = 0.5  # Attraction towards denser regions
    repulsion_radius = (
        0.05  # Distance for particle repulsion (as a fraction of the bound range)
    )
    repulsion_strength = 0.1  # Strength of the repulsive force.

    while eval_count < budget:
        for i in range(population_size):
            # 1. Diffusion: Randomly move the particle
            diffusion_vector = np.random.uniform(
                low=-diffusion_rate, high=diffusion_rate, size=dim
            ) * np.array([b[1] - b[0] for b in bounds])
            new_x = population[i] + diffusion_vector

            # 2. Aggregation: Move towards denser regions (other particles)
            attraction_vector = np.zeros(dim)
            for j in range(population_size):
                if i != j:
                    attraction_vector += (
                        population[j] - population[i]
                    )  # Attract to other particles
            new_x += (
                aggregation_strength * attraction_vector / (population_size - 1)
                if population_size > 1
                else 0
            )

            # 3. Repulsion: Avoid crowding
            repulsion_vector = np.zeros(dim)
            for j in range(population_size):
                if i != j:
                    distance = np.linalg.norm(population[j] - population[i])

                    # Normalize repulsion_radius to the dimension ranges
                    normalized_repulsion_radius = np.mean(
                        [repulsion_radius * (b[1] - b[0]) for b in bounds]
                    )

                    if distance < normalized_repulsion_radius:
                        repulsion_vector -= (population[j] - population[i]) / (
                            distance + 1e-8
                        )  # Repel from close particles, avoid div by 0

            new_x += repulsion_strength * repulsion_vector

            # 4. Boundary Handling
            new_x = np.clip(new_x, [b[0] for b in bounds], [b[1] for b in bounds])

            # 5. Evaluation and Update
            new_val = function(new_x)
            eval_count += 1

            if new_val < fitness[i]:
                fitness[i] = new_val
                population[i] = new_x.copy()

                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()

        # Adaptive Parameters (Optional - Adds Complexity and potential improvement, but increases code complexity)
        # Could adapt diffusion_rate and aggregation_strength based on population variance, similar to previous example.  Removing for conciseness.

        if eval_count >= budget:
            break

    return best_val, best_x
