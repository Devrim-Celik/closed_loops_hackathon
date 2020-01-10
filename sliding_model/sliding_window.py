import numpy as np
import concurrent.futures
import math
from auxiliary import *
from ode import ODE
from scipy import signal

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def error_function(Zb_dtdt, dt,  K=0.3):
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


def generate_windows(df, vel, fn, w_length=2.5, slide_step=1, dt=0.005, I_levels=9):
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
    dictionary['w_length'] = 2.5
    dictionary['s_step'] = 1
    dictionary['dt'] = 0.005
    dictionary['I_levels'] = 9

    initial_state = [0, 0, 0, 0]
    I = np.linspace(0, 2.0, I_levels)
    # for saving the values
    Best_I = np.zeros(len(df))

    # w = number of window
    print(len(df["time"]))
    for w in range(2000, 2020): #<============================================================== THIS IS NOT CORRECT I THINK
        print(w)
        # for erros and boundary values of different currents
        errors = [0]*len(I)
        boundaries = [False]*len(I)
        # candidate initial states
        candidate_init_states = [None]*len(I)


        # TODO do this using multiple threads in parallel
        #with concurrent.futures.ThreadPoolExecutor(max_workers=len(I)) as executor:
        #    executor.map(thread_I_fitness, I)

        # now go though every current and check which one is the fittest
        for indx_i, i in enumerate(I):
            Zb_dtdt_list = []
            Zt_dtdt_list = []
            for indx_within in range(w, w + int(w_length/dt)):
                # if this is the very first first step, use 0 are Zh_dt
                if w == 0 and indx_within == 0:
                    Zh_dt = 0
                # otherwise calculate it
                else:
                    Zh_dt = (df['profile'].iloc[w]-df['profile'].iloc[w-1])/dt


                # if this is the first step of a window, use initial_state
                # and safe the first step as the candidate for the initial
                # state of the next window (only need the first four arguments)
                if indx_within == w:
                    x = ODE(*initial_state, df['profile'].iloc[w], Zh_dt, i, dt)
                    candidate_init_states[indx_i] = x[:4]
                # otherwise the previous one
                else:
                    x = ODE(*x[:4], df['profile'].iloc[w], Zh_dt, i, dt)

                Zb_dtdt_list.append(x[4])
                Zt_dtdt_list.append(x[5])

            # TODO
            errors[indx_i] = error_function(Zb_dtdt_list, dt)
            boundaries[indx_i] = boundary_function(Zb_dtdt_list)

        # check if any does fulfill the boundary
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
        Best_I[w+1] = I[min_indx]
        initial_state = candidate_init_states[min_indx]

    return Best_I



if __name__=="__main__":
    generate_window_data()
    print("Done")
