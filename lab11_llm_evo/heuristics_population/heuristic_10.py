from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    This metaheuristic combines aspects of population-based search with the shrinking bound approach,
    using a diverse population that gradually focuses its search within tighter bounds around the best solutions found so far.
    """
    dimension = len(bounds)
    population_size = min(50, budget // 10)

    # Initialize population within bounds
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(
            low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=dimension
        )
        population.append(individual)

    # Evaluate initial population
    fitness = [function(individual) for individual in population]
    evaluations = population_size

    best_index = np.argmin(fitness)
    best_objective = fitness[best_index]
    best_solution = population[best_index].copy()

    # Main optimization loop
    while evaluations < budget:
        # Select parents based on fitness (tournament selection)
        selected_parents = []
        for _ in range(population_size):
            indices = np.random.choice(population_size, size=2, replace=False)
            if fitness[indices[0]] < fitness[indices[1]]:
                selected_parents.append(population[indices[0]])
            else:
                selected_parents.append(population[indices[1]])

        # Create offspring and shrink the bounds
        offspring = []
        for i in range(population_size):
            parent = selected_parents[i]
            # Shrink bounds around the current best solution
            shrinkage_factor = 0.5 * (
                1 - evaluations / budget
            )  # Shrinkage decreases linearly to 0
            local_bounds = [
                (
                    max(
                        bounds[j][0],
                        best_solution[j]
                        - shrinkage_factor * (bounds[j][1] - bounds[j][0]),
                    ),
                    min(
                        bounds[j][1],
                        best_solution[j]
                        + shrinkage_factor * (bounds[j][1] - bounds[j][0]),
                    ),
                )
                for j in range(dimension)
            ]

            # Generate offspring within the shrunk bounds using uniform sampling
            child = np.array(
                [np.random.uniform(low, high) for low, high in local_bounds]
            )
            offspring.append(child)

        # Evaluate offspring
        offspring_fitness = [function(child) for child in offspring]
        evaluations += population_size

        # Replacement (replace the worst with the best from the population and offspring)
        combined_population = population + offspring
        combined_fitness = fitness + offspring_fitness

        sorted_indices = np.argsort(combined_fitness)
        population = [combined_population[i] for i in sorted_indices[:population_size]]
        fitness = [combined_fitness[i] for i in sorted_indices[:population_size]]

        # Update best solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < best_objective:
            best_objective = fitness[best_index]
            best_solution = population[best_index].copy()

        if evaluations >= budget:
            break

    return best_objective, best_solution
