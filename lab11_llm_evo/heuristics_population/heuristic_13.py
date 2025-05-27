from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    This metaheuristic simulates a population of "ants" that probabilistically construct solutions, deposit "pheromone" on promising regions, and adjust their search based on pheromone levels and evaporation, dynamically balancing exploration and exploitation.
    """

    dimension = len(bounds)
    ant_count = min(40, budget // 10)  # Number of ants in the colony
    pheromone_deposit = 1.0  # Amount of pheromone deposited by each ant
    pheromone_evaporation = 0.1  # Rate at which pheromone evaporates
    alpha = 1.0  # Influence of pheromone on ant's decision
    beta = 2.0  # Influence of heuristic information on ant's decision
    initial_pheromone = 1e-6  # Avoid division by zero

    # Initialize pheromone trails
    pheromone = np.full(
        dimension, initial_pheromone
    )  # One pheromone value per dimension

    # Initialize best solution
    best_solution = None
    best_objective = float("inf")
    evaluations = 0

    # Helper function to evaluate a solution and increment the evaluation count
    def evaluate(solution):
        nonlocal evaluations
        evaluations += 1
        return function(solution)

    # Main loop
    while evaluations < budget:
        ant_solutions = []
        ant_objectives = []

        # Ants construct solutions
        for _ in range(ant_count):
            solution = np.zeros(dimension)
            for i in range(dimension):
                # Calculate probabilities based on pheromone and heuristic information
                pheromone_prob = pheromone[i] ** alpha
                range_val = bounds[i][1] - bounds[i][0]
                heuristic_value = (
                    range_val**beta
                )  # Heuristic value is based on bound size
                total = pheromone_prob + heuristic_value

                if total > 0:
                    prob_pheromone = pheromone_prob / total
                    prob_heuristic = heuristic_value / total

                    if np.random.rand() < prob_pheromone:
                        # Select a value based on pheromone in the dimension
                        solution[i] = np.random.uniform(
                            bounds[i][0], bounds[i][1]
                        )  # Explore
                    else:
                        # Exploit knowledge from heuristic
                        solution[i] = (
                            best_solution[i]
                            if best_solution is not None
                            else np.random.uniform(bounds[i][0], bounds[i][1])
                        )

                else:
                    solution[i] = np.random.uniform(bounds[i][0], bounds[i][1])

                solution[i] = np.clip(solution[i], bounds[i][0], bounds[i][1])

            objective = evaluate(solution)
            ant_solutions.append(solution)
            ant_objectives.append(objective)

            # Update best solution
            if objective < best_objective:
                best_objective = objective
                best_solution = solution.copy()

        # Update Pheromones
        for i in range(dimension):
            pheromone[i] *= 1 - pheromone_evaporation  # Evaporation

        # Deposit Pheromone on best solution
        for i in range(len(ant_solutions)):
            if ant_objectives[i] == best_objective:
                for j in range(dimension):
                    pheromone[j] += pheromone_deposit

        # Budget check
        if evaluations >= budget:
            break

    return best_objective, best_solution
