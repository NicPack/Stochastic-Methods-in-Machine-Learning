from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    A novel metaheuristic algorithm for minimizing a black-box function.

    This implementation uses a simplified version of differential evolution
    with some random restarts. It maintains a population of solutions
    and iteratively improves them using mutation, crossover, and selection.

    Args:
        function: The objective function to minimize.
        bounds:  A list of (lower, upper) pairs delimiting the search space.
        budget: The total number of objective-function evaluations allowed.

    Returns:
        A tuple containing the best objective value found and the
        corresponding decision vector.
    """

    dim = len(bounds)
    population_size = min(10 * dim, 50)  # Population size based on dimension

    # Initialize population randomly within bounds
    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(population_size, dim),
    )

    # Evaluate initial population
    fitness = np.array([function(x) for x in population])
    num_evaluations = population_size

    # Find the best individual in the initial population
    best_index = np.argmin(fitness)
    best_value = fitness[best_index]
    best_solution = population[best_index].copy()

    # Differential Evolution parameters (simplified)
    mutation_factor = 0.5
    crossover_probability = 0.7

    # Main optimization loop
    while num_evaluations < budget:
        for i in range(population_size):
            # Mutation: Select three random individuals (excluding the current one)
            indices = np.random.choice(population_size, 3, replace=False)
            if i in indices:
                indices = np.random.choice(population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(population_size, 3, replace=False)

            x1, x2, x3 = population[indices]

            # Create a mutant vector
            mutant = population[i] + mutation_factor * (x2 - x3)

            # Crossover: Create a trial vector
            trial = np.copy(population[i])
            for j in range(dim):
                if np.random.rand() < crossover_probability or j == np.random.randint(
                    0, dim
                ):  # Ensuring at least one param change
                    trial[j] = mutant[j]

            # Boundary handling (clip the values)
            trial = np.clip(trial, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate the trial vector
            trial_fitness = function(trial)
            num_evaluations += 1

            # Selection: Replace the parent if the trial vector is better
            if trial_fitness < fitness[i]:
                fitness[i] = trial_fitness
                population[i] = trial.copy()

                # Update the best solution found so far
                if trial_fitness < best_value:
                    best_value = trial_fitness
                    best_solution = trial.copy()

        # Random Restarts. Sometimes important to escape local optima and explore
        if (
            num_evaluations < budget and np.random.rand() < 0.01
        ):  # 1% Chance every generation to restart
            # Generate a new random population
            new_population = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
                size=(population_size, dim),
            )

            new_fitness = np.array([function(x) for x in new_population])
            num_evaluations += population_size

            # Replace population with new population and update best solution if better
            population = new_population
            fitness = new_fitness

            best_index = np.argmin(fitness)
            if fitness[best_index] < best_value:
                best_value = fitness[best_index]
                best_solution = population[best_index].copy()

        if num_evaluations >= budget:
            break

    return best_value, best_solution
