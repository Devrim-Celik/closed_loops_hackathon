###############################################################################
#
# Train a GA to optimize the weights of a neural network
#
# NOTE: this GA approach differs from others because it minimizes an objective
#       value instead of maximizing fitness.
#
###############################################################################

import os, time
import numpy as np
import matplotlib.pyplot as plt
import ga
import ann
from runProfile import calc_target
from multiprocessing import Pool, cpu_count

cores = cpu_count()

# GA Hyperparams
N_PARENTS = 40
N_GENERATIONS = 20
POPULATION_SIZE = 80
MUTATION_RATE = 1e-2

parrallel_processing = True

# ANN Hyperparams
ann_params = dict(input_shape=8, output_shape=1, l1_size=4, l2_size=4)

# Create initial random population of size POPULATION_SIZE
population = [ann.create_init_params(**ann_params) for i in range(POPULATION_SIZE)]

datadir = 'datasets/preproc/'
rnd_idx = np.random.choice(np.arange(104), cores)
datasets = np.array(sorted(os.listdir('datasets/preproc')))[rnd_idx]

# datasets = [
#     'ts1_1_k_3.0.csv',
#     # 'ts1_2_k_3.0.csv',
#     'ts1_3_k_3.0.csv',
#     'ts1_4_k_3.0.csv',
#     # 'ts2_k_20.0.csv',
#     # 'ts3_1_k_3.0.csv',
#     # 'ts3_2_k_3.0.csv',
#     # 'ts3_3_k_3.0.csv'
# ]

velocities = np.arange(20, 30, 2)

print('\nStarted Neural Network optimization using Genetic Algorithm.\n')
print('----------------------------')
print('Parameters:')
print('population size\t\t', POPULATION_SIZE)
print('no. of parents\t\t', N_PARENTS)
print('mutation rate\t\t', MUTATION_RATE)
print('no. of generations\t', N_GENERATIONS)
# print('velocities\t\t', velocities)
# print('no. of datasets\t\t', len(datasets))
print('----------------------------')

# Parallel processing loop
def thread_job(iter):
    targets_per_dataset = np.ones(POPULATION_SIZE)*9999
    dataset = iter[1]
    print('computing dataset', dataset)
    data = np.genfromtxt(datadir + dataset)
    K = float(dataset.split('.')[0][-1])

    for i, individual in enumerate(iter[0]):
        percentage = (i/POPULATION_SIZE)*100
        if percentage % 10 == 0:
            print('[ {0:.0f}% ] dataset {1}'.format(percentage, dataset))
        # Calculate targets of individuals
        individual_target = calc_target(
            ann.network_function(individual), data, K)
        targets_per_dataset[i] = individual_target
    print('[ DONE ] dataset {}'.format(dataset))
    return targets_per_dataset

# Optimize population
avg_targets, best_targets = [], []
for generation in range(N_GENERATIONS):
    start = time.time()
    print('\n%%%%%%%%%%%%%%%%%%%%')
    print('%  Generation\t', generation)
    print('%%%%%%%%%%%%%%%%%%%%')

    if parrallel_processing:
        iterable = [(population, dataset) for dataset in datasets]
        pool = Pool(processes=cores)
        targets = pool.map(thread_job, iterable)
        pool.close()

        targets = np.array(targets)
        targets = np.sum(targets, axis =0)
        targets = targets / len(datasets)

    else:
        # Loop through data
        targets_per_dataset = np.zeros(POPULATION_SIZE)
        for dataset in datasets:
            print('\nData:', dataset)
            targets_per_velocity = np.zeros(POPULATION_SIZE)
            for velocity in velocities:
                print('Velocity =', velocity)
                f = dataset.split('.')
                k = float(f[0].split('_')[-1])
                data = np.genfromtxt(datadir + '{}.{}_vel_{}.csv'.format(f[0], f[1], velocity))
                    # '.'.join([f[0],f[1],f'vel_{velocity}',f[2]]))
                for i, individual in enumerate(population):
                    # Calculate targets of individuals
                    individual_target = calc_target(
                        ann.network_function(individual), data, k)
                    targets_per_velocity[i] += individual_target
                    # print('Target of Individual =', individual_target)
            # Add average across all velocities
            targets_per_dataset += targets_per_velocity / len(velocities)
        # Add average across all datasets
        print()
        targets = targets_per_dataset / len(datasets)
        print('Individuals =', targets)

    # Convert all individuals to vector instead of matrix
    population = [ann.weights_to_vec(individual) for individual in population]
    population = np.array(population)

    # Compute stats
    avg_targets.append(sum(targets)/float(len(targets)))
    best_targets.append(min(targets))
    print("Best\t", best_targets[-1])
    print("Worst\t", max(targets))
    print("Mean\t", avg_targets[-1])
    print("Variance", np.var(targets))
    print()

    # Compute next generation
    parents = ga.select_mating_pool(population, targets, N_PARENTS)
    offspring = ga.crossover(parents,
        offspring_size=(population.shape[0] - parents.shape[0],
                        population.shape[1]))
    offspring = ga.mutate(offspring, MUTATION_RATE)
    population[:N_PARENTS] = parents
    population[N_PARENTS:] = offspring

    # Put back into matrix form
    population = [ann.vec_to_mat(**ann_params, vec=vec) for vec in population]
    time_passed = time.time()-start
    print('Generation took {0:.4f} seconds.'.format(time_passed))

# Plot average target values
plt.plot(avg_targets, c='red', label='average of population')
plt.plot(best_targets, c='orange', label='best individual in population')
plt.xlabel('Generation')
plt.ylabel('Target Value')
# plt.xticks(np.arange(0, N_GENERATIONS+1, 100))
plt.legend()
plt.show()
