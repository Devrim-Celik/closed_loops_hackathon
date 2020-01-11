import numpy as np
import concurrent.futures
import math
from auxiliary import *
from ode import ODE
from scipy import signal

import time as time_module
from multiprocessing.pool import Pool
import multiprocessing

cores = multiprocessing.cpu_count()

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def error_function(Zb_dtdt, dt,  K=3):
    """
    Calculate error value. We want to minimize. Given by
        T = alpha_1 + K*alpha_2

        where alpha_1 and alpha_2 are frequency bandpasses
    """
    #TODO Get K here from file name
    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = butter_bandpass(0.4, 3, int(1/dt), 2)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    #calculate variance alpha_1
    varZb_dtdt=np.var(z)

    #compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = butter_bandpass(10, 30, int(1/dt), 1)
    zi = signal.lfilter_zi(b, a)
    z1, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)
    #calculate variance alpha_2
    varZb_dtdt_h=np.var(z1)

    #compute T_target
    return K*varZb_dtdt_h + varZb_dtdt


def boundary_function(Zt_dtdt, Mb = 500, Mt = 50):
    """
    Function for calculating whether boundary condition was hold or not.

    Args
        Zt_dtdt (float): acceleration of tire
        Mb (float): mass quarter body [kg]
        Mt (float): mass tire + syspention system [kg]
    Returns
        (boolean)
    """

    #standard deviation of Zt_dtdt
    devZt_dtdt = np.std(Zt_dtdt*Mt)

    #boundary condition
    F_stat_bound = (Mb+Mt)*9.81/3.0
    if devZt_dtdt > (F_stat_bound):
        return False
    else:
        return True

def generate_windows(df, vel, fn, w_length=2.5, slide_step=1, dt=0.005, K=3, I_levels=9):
    #TODO did not implement for higher slide_step than 1
    """

    Function that generates multiple time windows for a given time series. Furthermore
    saves them as json files.

    Args:
        df                  data frame with cols: time, trip and profile
        vel                 velocity
        fn                  file_name
        w_length            length of a window in seconds
        slide_step          offset of different windows
        dt                  timesteps
        I_levels            number of current levels
    """
    #window length has to be in seconds --> 2.5 secondshttps://imgflip.com/i/3lrsm9
    dictionary = {}
    dictionary['vel'] = vel
    dictionary['fn'] = fn
    dictionary['w_length'] = w_length
    dictionary['s_step'] = slide_step
    dictionary['dt'] = dt
    dictionary['I_levels'] = I_levels

    # initial state for the ODE
    initial_state = [0, 0, 0, 0]
    # level of currents we are testint
    I = np.linspace(0, 2.0, I_levels)
    # for saving the values, -1 indicates unassigned
    optimal_I = (-1)*np.ones(len(df))
    # for saving the erros along the way
    final_erros = np.zeros(len(df))
    # total time steps steps
    N = len(df) 

    # steps per window
    n = int(w_length/dt)
    # beginning and end edge length is half the window
    if n % 2 == 0:
        k = int(n/2)
    else:
        k = int((n-1)/2)

    # first index that is to be assigned by every window
    to_be_assigned_indx = k+1

    #TODO for now set edges zero:
    optimal_I[:k] = 0
    optimal_I[-k:] = 0

    print("N={}, n={}, k={}, to_be_assigned_indx={}".format(N, n, k, to_be_assigned_indx))

    # w indicates the start of every window, iterating by slide_step
    for w in range(0, N-2*k, slide_step):
        print("[*] Window [{:7d}] of [{:7d}].".format(w, int((N-2*k)/slide_step)))

        # for errors and boundary values of different currents
        # candidate initial states so we can take over the ODE
        errors = [0]*len(I)
        boundaries = [False]*len(I)
        candidate_init_states = [None]*len(I)

        # now go though every current and check which one is the fittest
        # for indx_i, i in enumerate(I):
        iterable = [(indx, i) for indx, i in enumerate(I)]
        print('\n%%Current window: ', w)
        start = time_module.time()
        pool = Pool(processes=cores - 1)
        pool.map(check_fittest, iterable)  # vel is the iterable parameter
        pool.close()
        print(f'Took {time_module.time() - start:.3f} seconds to process window.')

        # pool.join() # wait until all jobs are finished to finish the pool of jobs
        def check_fittest(iter):
            # for each current we reset those lists; they used to calculate
            # loss and boundary condition
            indx_i = iter[0]
            i = iter[1]
            Zb_dtdt_list = []
            Zt_dtdt_list = []
            # here you take steps through the window
            for indx_within in range(w, w + n):
                # if Best_I already is asigned (i.e. edge value), use this
                # value instead of the candidate i
                if optimal_I[indx_within] != -1:
                    used_i = optimal_I[indx_within]
                else:
                    used_i = i
                #print(used_i)
                # if this is the very first first step, ts2_k_20.0.csvuse 0 as Zh_dt
                if w == 0 and indx_within == 0:
                    Zh_dt = 0
                # otherwise calculate it
                else:
                    Zh_dt = (df['profile'].iloc[indx_within]-df['profile'].iloc[indx_within-1])/dt


                # if this is the first step of a window, use initial_state
                if indx_within == w:
                    x = ODE(*initial_state, df['profile'].iloc[indx_within], Zh_dt,
                        used_i, dt)
                # otherwise the previous one
                else:
                    x = ODE(*x[:4], df['profile'].iloc[indx_within], Zh_dt, used_i, dt)

                # from one of the candidate currents, we will need one
                # paricular state as the initial state for the next window
                # if the window indx is (slide_step-1) away from the start
                # of the window, the resulting state will be saved
                if indx_within == w + (slide_step-1):
                    candidate_init_states[indx_i] = x[:4]

                # append those for later error calculations
                Zb_dtdt_list.append(x[4])
                Zt_dtdt_list.append(x[5])

            # here we calculate error and boundary
            errors[indx_i] = error_function(Zb_dtdt_list, dt, K=K)
            boundaries[indx_i] = boundary_function(Zb_dtdt_list)

        print(errors)
        print(boundaries)

        if boundaries == [False]*len(I):
            raise Exception('\n[!]No current fulfilled boundary!\n')

        min = math.inf

        for indx, (e, b) in enumerate(zip(errors, boundaries)):
            # go through all currents, and check if they are the smalles given
            # that they fulfill the boundary
            if b and e < min:
                min = e
                min_indx = indx
        # get the best current and the corresponding initial state
        for ite in range(slide_step):
            optimal_I[w+to_be_assigned_indx] = I[min_indx]
            final_erros[w+to_be_assigned_indx] = min
        initial_state = candidate_init_states[min_indx]

    return optimal_I, final_erros



if __name__=="__main__":
    generate_window_data()
    print("Done")
