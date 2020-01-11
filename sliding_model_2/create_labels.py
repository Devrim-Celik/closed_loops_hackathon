from main import run
import os
from multiprocessing import Pool, cpu_count
import time as time_module
import numpy as np

cores = cpu_count()

datadir = 'datasets/preproc/'
datasets = os.listdir(datadir)
velocities = np.arange(5, 31, 1)
window_sizes = [50, 100, 250, 500, 1000, 2500, 5000]

def thread_job(iter):
    for window_size in window_sizes:
        run(datadir, dataset=iter[0], velocity=iter[1], win_size=window_size, save=True, plot=False)

for dataset in datasets:
    print('\n\n\n%% Processing dataset', dataset)
    start = time_module.time()
    iterable = [(dataset, vel) for vel in velocities]
    pool = Pool(processes=cores)
    pool.map(thread_job, iterable)
    pool.close()
    print(f'\n\n\nTook {time_module.time() - start:.3f} seconds to process dataset.\n\n\n')
