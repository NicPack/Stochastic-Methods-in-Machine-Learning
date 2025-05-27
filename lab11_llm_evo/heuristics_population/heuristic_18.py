from typing import Callable, List, Tuple

import numpy as np


def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int,
) -> Tuple[float, np.ndarray]:
    """
    This metaheuristic combines a global, gradient-free random search with a biased sampling toward the best-performing region, adaptively adjusting the sampling distribution's variance based on observed fitness variance.
    """

    dimension = len(bounds)
    initial_samples = min(100, budget // 10)
    adaptive_iterations = 5

    # 1. Initial Global Search
    samples = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(initial_samples, dimension),
    )
    fitness = np.array([function(x) for x in samples])
    evaluations = initial_samples

    best_index = np.argmin(fitness)
    best_objective = fitness[best_index]
    best_solution = samples[best_index].copy()

    # 2. Adaptive Iterations
    while evaluations < budget:
        # 3. Calculate Variance of Samples in Each Dimension
        variances = np.var(samples, axis=0)

        # 4. Adjust standard deviation to sampling, scaled to bound ranges
        std_devs = np.sqrt(variances)
        bound_ranges = np.array([b[1] - b[0] for b in bounds])
        adaptive_std_devs = (
            std_devs * bound_ranges / np.mean(bound_ranges)
        )  # Scale by the mean bound range

        # 5. Generate New Samples Based on Best Solution and Adaptive Standard Deviations
        new_samples = []
        for _ in range(adaptive_iterations):
            new_sample = np.random.normal(loc=best_solution, scale=adaptive_std_devs)

            # Clip to bounds
            for j in range(dimension):
                new_sample[j] = np.clip(new_sample[j], bounds[j][0], bounds[j][1])
            new_samples.append(new_sample)
        new_samples = np.array(new_samples)

        # 6. Evaluate New Samples
        new_fitness = np.array([function(x) for x in new_samples])
        evaluations += adaptive_iterations

        # 7. Combine Old and New Samples
        all_samples = np.vstack((samples, new_samples))
        all_fitness = np.concatenate((fitness, new_fitness))

        # 8. Select Best Samples for Next Iteration
        sorted_indices = np.argsort(all_fitness)[:initial_samples]
        samples = all_samples[sorted_indices]
        fitness = all_fitness[sorted_indices]

        # 9. Update Best Solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < best_objective:
            best_objective = fitness[best_index]
            best_solution = samples[best_index].copy()

        if evaluations >= budget:
            break

    return best_objective, best_solution
