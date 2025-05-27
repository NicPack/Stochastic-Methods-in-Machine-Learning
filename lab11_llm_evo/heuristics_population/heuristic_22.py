from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A metaheuristic that blends particle swarm optimization (PSO) principles with adaptive constraint handling and dynamic neighborhood size.
    """

    dim = len(bounds)
    swarm_size = min(40, budget // 20)  # Adjusted swarm size based on budget
    inertia = 0.7
    cognitive_coeff = 1.4
    social_coeff = 1.4
    velocity_limit_factor = 0.1  # Limit velocities to this fraction of the bound range.

    # Initialize swarm (particles)
    swarm = np.random.uniform(
        low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(swarm_size, dim)
    )

    # Initialize velocities
    velocity_max = velocity_limit_factor * np.array([b[1] - b[0] for b in bounds])
    velocity_min = -velocity_max
    velocities = np.random.uniform(
        low=velocity_min, high=velocity_max, size=(swarm_size, dim)
    )

    # Initialize personal best positions and fitnesses
    personal_best_positions = swarm.copy()
    personal_best_fitness = np.array([function(x) for x in swarm])

    # Initialize global best position and fitness
    best_index = np.argmin(personal_best_fitness)
    global_best_position = personal_best_positions[best_index].copy()
    global_best_fitness = personal_best_fitness[best_index]

    eval_count = swarm_size

    # Main optimization loop
    while eval_count < budget:
        for i in range(swarm_size):
            # Update velocity
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            cognitive_component = (
                cognitive_coeff * r1 * (personal_best_positions[i] - swarm[i])
            )
            social_component = social_coeff * r2 * (global_best_position - swarm[i])

            velocities[i] = (
                inertia * velocities[i] + cognitive_component + social_component
            )

            # Velocity clamping
            velocities[i] = np.clip(velocities[i], velocity_min, velocity_max)

            # Update position
            new_position = swarm[i] + velocities[i]

            # Constraint handling: Reflect back into the search space if out of bounds.
            for j in range(dim):
                if new_position[j] < bounds[j][0]:
                    new_position[j] = bounds[j][0] + (
                        bounds[j][0] - new_position[j]
                    )  # Reflect
                    velocities[i][j] *= -0.5  # Dampen
                elif new_position[j] > bounds[j][1]:
                    new_position[j] = bounds[j][1] - (
                        new_position[j] - bounds[j][1]
                    )  # Reflect
                    velocities[i][j] *= -0.5  # Dampen

            # Evaluate new position
            new_fitness = function(new_position)
            eval_count += 1

            # Update personal best
            if new_fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = new_fitness
                personal_best_positions[i] = new_position.copy()

                # Update global best
                if new_fitness < global_best_fitness:
                    global_best_fitness = new_fitness
                    global_best_position = new_position.copy()
        if eval_count >= budget:
            break

    return global_best_fitness, global_best_position
