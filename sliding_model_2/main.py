
import numpy as np
import matplotlib.pyplot as plt
from runProfile import calc_target


datadir = 'datasets/preproc/'
dataset = 'ts3_2_k_3.0.csv'

# Set the velocity
velocity = 6

# Read the data
f = dataset.split('.')
K = float(f[0].split('_')[-1])
profile = np.genfromtxt(datadir + f'{f[0]}.{f[1]}_vel_{velocity}.csv')

state = [0, 0, 0, 0, 0, 0] # initial state
N = len(profile) # length of profile
w_size = 500 # size of the window
possible_Is = np.linspace(0.0, 2, 20)
optimal_I = np.zeros(N)

best_targets = []
for idx in range(0, N, w_size):

    w_start, w_end = idx, idx+w_size
    if w_end > N:
        break

    window_profile = profile[w_start:w_end]

    targets, states = [], []

    for i in possible_Is:
        target, next_state = calc_target(state, window_profile, i, K)
        targets.append(target)
        states.append(next_state)

    print(targets)
    best_targets.append(min(targets))
    min_target_index = targets.index(min(targets))
    state = states[min_target_index]
    optimal_I[w_start:w_end] = possible_Is[min_target_index]

print('\nBest targets', best_targets)
# print('Average target', sum(best_targets)/idx)

# Test optimal_I against constants
print()
print('Optimal I target:', calc_target([0,0,0,0,0,0], profile, optimal_I, K)[0])
for i in possible_Is:
    print(f'Const {i:.3f} target:', calc_target([0,0,0,0,0,0], profile, i, K)[0])

plt.plot(profile*100)
plt.plot(optimal_I)
plt.show()
