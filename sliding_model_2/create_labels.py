from main import run
import os
from multiprocessing import Pool, cpu_count
import time as time_module
import numpy as np

cores = cpu_count()

datadir = 'datasets/final_preproc/'
datasets = sorted(os.listdir(datadir))
window_sizes = [50, 100, 120, 150, 170, 250, 500, 750, 1000, 1750, 2500]

def thread_job(iter):
    run(datadir, dataset=iter[0], win_size=iter[1], save=True, plot=False)
    print('Finished win_size {}'.format(iter[1]))

for dataset in datasets:
    print('\n%% Processing dataset', dataset)
    start = time_module.time()
    iterable = [(dataset, win_size) for win_size in window_sizes]
    pool = Pool(processes=cores)
    pool.map(thread_job, iterable)
    pool.close()
    print('Took {0:.3f} seconds to process dataset.'.format(time_module.time() - start))
