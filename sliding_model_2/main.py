
import numpy as np
import matplotlib.pyplot as plt
from runProfile import calc_target
import sys, argparse, time


def run(datadir, dataset, win_size, plot=False, save=False):

    print('Computing {} with window size {}'.format(dataset, win_size))

    # Read the data
    K = float(dataset.split('.')[0][-1])
    profile = np.genfromtxt(datadir + dataset)

    state = [0, 0, 0, 0, 0, 0] # initial state
    N = len(profile) # length of profilef
    possible_Is = np.array([0.,.05,.1,.15,.2,.25,.3,.4,.5,.6,.9,1.2,1.5,1.8])
    optimal_I = np.zeros(N)

    best_targets = []
    for idx in range(0, N, win_size):

        w_start, w_end = idx, idx+win_size
        if w_end > N:
            break

        window_profile = profile[w_start:w_end]

        targets, states = [], []

        for i in possible_Is:
            target, next_state = calc_target(state, window_profile, i, K)
            targets.append(target)
            states.append(next_state)

        # print(targets)
        best_targets.append(min(targets))
        min_target_index = targets.index(min(targets))
        state = states[min_target_index]
        optimal_I[w_start:w_end] = possible_Is[min_target_index]

    # print('\nBest targets', best_targets)
    # print('Average target', sum(best_targets)/idx)

    # Test optimal_I against constants
    optimal_I_error = calc_target([0,0,0,0,0,0], profile, optimal_I, K)[0]
    # print('Optimal I target:', optimal_I_error)
    # for i in possible_Is:
    #     print(f'Const {i:.3f} target:', calc_target([0,0,0,0,0,0], profile, i, K)[0])

    if plot:
        plt.plot(profile*100)
        plt.plot(optimal_I)
        plt.show()

    if save:
        import csv
        fname = dataset[:-4]
        fname = 'optimal_I_{0}_wsize_{1}_err_{2:.7f}.csv'.format(fname, win_size, optimal_I_error)
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
    dataset = 'ts1_3_k_3.0_vel_5.csv'
    win_size = 500
    save = False

    run(datadir, dataset, win_size, save=save)

    sys.exit()
