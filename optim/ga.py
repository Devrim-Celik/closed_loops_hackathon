# The genetic algorithm functions

import numpy as np

def select_mating_pool(population, fitnesses, n_parents):
    '''
    Get the <n_parents> fittest individuals.
    '''
    # Alternative: use the lottery like selection from the COREM GA
    individual_shape = population.shape[1]
    parents = np.empty((n_parents, individual_shape))
    for i in range(n_parents):
        max_fitness_idx = np.where(fitnesses==np.min(fitnesses))[0][0]
        parents[i, :] = population[max_fitness_idx, :]
        fitnesses[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents.
    # Here it is at the center; could also be random
    crossover_point = np.uint32(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0] # Index of the first parent to mate.
        parent2_idx = (k+1) % parents.shape[0] # Index of the second parent to mate.
        # First half taken from the first parent.
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        # Second half taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutate(offspring, mutation_rate):
    # How many genes to mutate per individual
    num_mutations = np.uint32(mutation_rate * offspring.shape[1])
    # Mutation changes a single gene in each offspring randomly.
    for i in range(offspring.shape[0]):
        mutation_idx = np.array(random.sample(range(0, offspring.shape[1]), num_mutations))
        # Add random value to the individual.
        offspring[i, mutation_idx] += np.random.uniform(-1.0, 1.0, 1)
    return offspring
