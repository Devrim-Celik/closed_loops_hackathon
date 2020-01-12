from main import run
import os
from multiprocessing import Pool, cpu_count
import time as time_module
import numpy as np

cores = cpu_count()

datadir = 'datasets/preproc/'
datasets = sorted(os.listdir(datadir))
window_sizes = [50, 100, 250, 500, 1000, 2500, 5000]

def thread_job(iter):
    print('\n%% Processing dataset', iter)
    for win_size in window_sizes:
        run(datadir, dataset=iter, win_size=win_size, save=True, plot=False)
    print('Took {0:.3f} seconds to process dataset.'.format(time_module.time() - start))

start = time_module.time()
pool = Pool(processes=cores)
pool.map(thread_job, datasets)
pool.close()
