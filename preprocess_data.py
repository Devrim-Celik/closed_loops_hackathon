from auxiliary import time_and_profile
import csv

import time as time_module
from multiprocessing.pool import Pool
import multiprocessing

cores = multiprocessing.cpu_count()

profileLocation = 'datasets/'
# profile filenames
datasets = [
    'ts1_1_k_3.0.csv',
    'ts1_2_k_3.0.csv',
    'ts1_3_k_3.0.csv',
    'ts1_4_k_3.0.csv',
    'ts2_k_20.0.csv',
    'ts3_1_k_3.0.csv',
    'ts3_2_k_3.0.csv',
    'ts3_3_k_3.0.csv'
]
velocities = [i for i in range(5, 31)]  # loop over all velocities


def thread_job(iter):
    fname = iter[0]
    vel = iter[1]
    time, profile = time_and_profile(profileLocation + fname, vel)
    with open(profileLocation + "preproc2/" + fname[:-4] + "_" + str(vel) + ".csv", mode='w') as file:
        writer = csv.writer(file, delimiter=',')
        for p in profile:
            writer.writerow([p])


for dataset in datasets:
    print('\n%% Processing dataset', dataset)
    fname = profileLocation + dataset
    start = time_module.time()
    for vel in range(5, 31):  # loop over all velocities
        print('… velocity', vel)
        time, profile = time_and_profile(fname, vel)
        with open(profileLocation + "preproc/" + dataset[:-4] + "_vel_" + str(vel) + ".csv", mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            for p in profile:
                writer.writerow([p])
    print(f'Took {time_module.time()-start:.3f} seconds to process dataset.')
    threads = list()
    iterable = [(dataset, vel) for vel in velocities]
    with Pool(processes=cores-2) as pool:
        pool.map(thread_job, iterable)  # vel is the iterable parameter
        # pool.close()
        # pool.join()
