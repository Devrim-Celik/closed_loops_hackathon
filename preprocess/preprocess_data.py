from auxiliary import time_and_profile
import csv

import time as time_module
from multiprocessing.pool import Pool
import multiprocessing

cores = multiprocessing.cpu_count()

profileLocation = 'datasets/'
# profile filenames
# datasets = [
#     'ts1_1_k_3.0.csv',
#     'ts1_2_k_3.0.csv',
#     'ts1_3_k_3.0.csv',
#     'ts1_4_k_3.0.csv',
#     'ts2_k_20.0.csv',
#     'ts3_1_k_3.0.csv',
#     'ts3_2_k_3.0.csv',
#     'ts3_3_k_3.0.csv'
# ]
# velocities = [i for i in range(5, 31)]  # loop over all velocities

datasets = ['ts4_k_20.0.csv']
velocities = [8.3, 13.8, 19.4, 27.7]


def thread_job(iter):
    fname = iter[0]
    vel = iter[1]
    print('… velocity', vel)
    time, profile = time_and_profile(profileLocation + fname, vel)
    with open(profileLocation + "final_preproc/" + fname[:-4] + "_vel_" + str(vel) + ".csv", mode='w') as file:
        writer = csv.writer(file, delimiter=',')
        for p in profile:
            writer.writerow([p])


for dataset in datasets:
    print('\n%% Processing dataset', dataset)
    fname = profileLocation + dataset
    start = time_module.time()
    iterable = [(dataset, vel) for vel in velocities]
    pool = Pool(processes=cores - 2)
    pool.map(thread_job, iterable)  # vel is the iterable parameter
    pool.close()
    print(f'Took {time_module.time() - start:.3f} seconds to process dataset.')
    # pool.join() # wait until all jobs are finished to finish the pool of jobs
