"""Here's a design and implementation of a new metaheuristic algorithm based on the provided ones, adhering to your specifications.

**1. Common Idea:**

All the provided heuristics share the concept of iteratively refining a solution or a population of solutions within the given bounds, balancing exploration (searching broadly) and exploitation (focusing on promising regions). They all use function evaluations to guide their search process. Most of them have populations, update rules, and mechanisms to escape local optima.

**2. New Heuristic Description:**

The new metaheuristic combines elements of population-based search, gradient estimation (finite differences), and adaptive step size control, iteratively refining solutions by approximating the local gradient and adjusting movement based on success and failure.

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
    Combines population-based search, gradient estimation (finite differences), and adaptive step size control.
    """
    dimension = len(bounds)
    population_size = min(20, budget // 50)  # Adjusted population size
    step_size = 0.1  # Initial step size (as a fraction of the bounds)
    success_rate = 0.5  # Target success rate
    learning_rate = 0.1  # Learning Rate

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

    archive_solutions = [best_solution.copy()]
    archive_fitness = [best_objective]

    # Main optimization loop
    while evaluations < budget:
        # Iterate through each individual in the population
        for i in range(population_size):
            current_solution = population[i].copy()
            current_fitness = fitness[i]

            # Estimate gradient using finite differences
            gradient = np.zeros(dimension)
            for j in range(dimension):
                # Calculate a small perturbation
                perturbation = np.zeros(dimension)
                perturbation[j] = step_size * (bounds[j][1] - bounds[j][0])

                # Evaluate the function at the perturbed points
                x_plus = current_solution + perturbation
                x_minus = current_solution - perturbation

                # Clip to respect bounds
                x_plus = np.clip(x_plus, [b[0] for b in bounds], [b[1] for b in bounds])
                x_minus = np.clip(
                    x_minus, [b[0] for b in bounds], [b[1] for b in bounds]
                )

                fitness_plus = function(x_plus)
                fitness_minus = function(x_minus)
                evaluations += 2  # Count 2 function calls

                # Approximate Gradient
                gradient[j] = (
                    (fitness_minus - fitness_plus) / (2 * perturbation[j])
                    if perturbation[j] != 0
                    else 0
                )

            # Move in the direction of the negative gradient
            new_solution = current_solution - step_size * gradient

            # Clip to respect bounds
            new_solution = np.clip(
                new_solution, [b[0] for b in bounds], [b[1] for b in bounds]
            )

            # Evaluate the new solution
            new_fitness = function(new_solution)
            evaluations += 1

            # Adaptive Step Size Control
            if new_fitness < current_fitness:
                # Successful move
                population[i] = new_solution.copy()
                fitness[i] = new_fitness

                # Learning rate based adaptation
                step_size *= 1 + learning_rate  # increase step size

                # Update best solution
                if new_fitness < best_objective:
                    best_objective = new_fitness
                    best_solution = new_solution.copy()
                    archive_solutions.append(best_solution.copy())
                    archive_fitness.append(best_objective)

            else:
                # Unsuccessful Move
                step_size *= 1 - learning_rate  # decrease step size

            step_size = np.clip(step_size, 1e-6, 0.2)  # Clamp step size

            if evaluations >= budget:
                break
        if evaluations >= budget:
            break

    return best_objective, best_solution
