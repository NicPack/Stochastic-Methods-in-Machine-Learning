import numpy as np
from typing import Callable, List, Tuple

def new_metaheuristic(
	function: Callable[[np.ndarray], float], 
    bounds: List[Tuple[float, float]], 
    budget: int
) -> Tuple[float, np.ndarray]:
    """
    A novel metaheuristic algorithm for minimizing a black-box function 
    subject to bound constraints.  This implements a simplified evolutionary 
    strategy with a population and mutation.

    Args:
        function: The objective function to minimize.
        bounds: A list of (lower, upper) bounds for each dimension.
        budget: The total number of function evaluations allowed.

    Returns:
        A tuple containing the best objective value found and the 
        corresponding decision vector.
    """

    dimension = len(bounds)
    population_size = min(100, budget // 10) # Adjust population size based on budget
    
    # Initialize the population
    population = []
    for _ in range(population_size):
        individual = np.random.uniform(
            low=[b[0] for b in bounds], 
            high=[b[1] for b in bounds], 
            size=dimension
        )
        population.append(individual)

    # Evaluate initial population
    fitness = [function(individual) for individual in population]
    evaluations = population_size # initial evaluations used

    best_index = np.argmin(fitness)
    best_objective = fitness[best_index]
    best_solution = population[best_index]

    # Main optimization loop
    while evaluations < budget:
        # Selection (simple tournament selection - select two random individuals)
        indices = np.random.choice(population_size, size=2, replace=False)
        parent1_index = indices[0]
        parent2_index = indices[1]

        if fitness[parent1_index] < fitness[parent2_index]:
            parent = population[parent1_index]
        else:
            parent = population[parent2_index]

        # Mutation (Gaussian mutation with adaptive step size)
        mutation_rate = 0.1 # Probability of mutation for each dimension.
        mutation_scale = (np.max([b[1] - b[0] for b in bounds]) / 100) # Scale of mutation adjusted relative to range of the search space

        mutated_individual = parent.copy()
        for i in range(dimension):
            if np.random.rand() < mutation_rate:
                mutated_individual[i] += np.random.normal(0, mutation_scale)
                # Clip to bounds
                mutated_individual[i] = np.clip(mutated_individual[i], bounds[i][0], bounds[i][1])

        # Evaluate the offspring
        mutated_objective = function(mutated_individual)
        evaluations += 1

        # Replacement (elitist replacement - replace worst in the population if offspring is better)
        worst_index = np.argmax(fitness)
        if mutated_objective < fitness[worst_index]:
            population[worst_index] = mutated_individual
            fitness[worst_index] = mutated_objective

            # Update best solution if necessary
            if mutated_objective < best_objective:
                best_objective = mutated_objective
                best_solution = mutated_individual
                
        if evaluations >= budget:
            break

    return best_objective, best_solution