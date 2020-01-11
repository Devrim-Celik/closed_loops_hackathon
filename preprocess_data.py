from auxiliary import time_and_profile
import csv
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
    threads = list()
    iterable = [(dataset, vel) for vel in velocities]
    with Pool(processes=cores) as pool:
        pool.map(thread_job, iterable)  # vel is the iterable parameter
        pool.close()
        pool.join()
