
import numpy as np
import matplotlib.pyplot as plt
from runProfile import calc_target
import sys, argparse


def run(datadir, dataset, velocity=20, win_size=500, plot=False, save=False):

    # Read the data
    K = float(dataset.split('.')[0][-1])
    profile = np.genfromtxt(datadir + dataset)

    state = [0, 0, 0, 0, 0, 0] # initial state
    N = len(profile) # length of profile
    w_size = win_size # size of the window
    possible_Is = np.array([0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.,1.2,1.4,1.6,1.8])
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
    optimal_I_error = calc_target([0,0,0,0,0,0], profile, optimal_I, K)[0]
    print('Optimal I target:', optimal_I_error)
    for i in possible_Is:
        print(f'Const {i:.3f} target:', calc_target([0,0,0,0,0,0], profile, i, K)[0])

    if plot:
        plt.plot(profile*100)
        plt.plot(optimal_I)
        plt.show()

    if save:
        import csv
        fname = dataset.split('.')[0]
        fname = f'optimal_I_{fname}_vel_{velocity}_wsize_{w_size}_err_{optimal_I_error:.7f}.csv'
        with open("datasets/labels/"+fname, mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            for i in optimal_I:
                writer.writerow([i])


# def getArgs(args=sys.argv[1:]):
#     """
#     Parses arguments from command line.
#     """
#     parser = argparse.ArgumentParser(description="Parses command.")
#     parser.add_argument("--dataset", type=str, help="Which dataset to use.")
#     parser.add_argument("--velocity", type=int,
#         help="The velocity to run the simulation at.")
#     parser.add_argument("--plot", type=bool, action='store_true', help="Plot results.")
#     parser.add_argument("--save", type=bool, help="Save results to storage.")
#     return parser.parse_args(args)

if __name__ == '__main__':


    datadir = 'datasets/preproc/'
    dataset = 'ts3_2_k_3.0.csv'
    velocity = 5
    win_size = 5000
    save = True

    run(datadir, dataset, velocity, win_size, save=save)

    sys.exit()
