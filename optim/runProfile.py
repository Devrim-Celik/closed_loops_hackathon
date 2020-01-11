import csv
from ODE import ODE
import numpy as np
from scipy import signal

# simulation interval in seconds
dt = 0.005

# initial suspension states
initialSuspensionState = np.zeros(6)  # initial suspension states at t = 0
initialSuspensionState[0] = 0  # Zb: z-position body [m]
initialSuspensionState[1] = 0  # Zt: z-position tire [m]
initialSuspensionState[2] = 0  # Zb_dt: velocity body in z [m/s]
initialSuspensionState[3] = 0  # Zt_dt: velocity tire in z [m/s]
initialSuspensionState[4] = 0  # Zb_dtdt: acceleration body in z [m/s^2]
initialSuspensionState[5] = 0  # Zt_dtdt: acceleration tire in z [m/s^2]

def calc_target(ind, profile, K):
    '''
    ind: individual from a population (GA)
    return: target
    '''

    def solver_dampingForce(intialSuspensionState, profile, last_timestep=None):
        '''
        intialSuspensionState: [Zb, Zt, Zb_dt, Zt_dt] at t=0
        H: road profile [m] over time intervals
        t: time intervals
        '''
        if not last_timestep:
            N = len(profile)
        else:
            N = last_timestep - 1
        x = np.zeros((N + 1, 6))  # array of [Zb, Zt, Zb_dt, Zt_dt, Zb_dtdt, Zt_dtdt] at each time interval
        # Set inital state of the system and initial profile
        x[0] = intialSuspensionState  # [Zb, Zt, Zb_dt, Zt_dt, Zb_dtdt, Zt_dtdt] at t=0
        Zh, Zh_dt = profile[0], 0.
        for n in range(0, N-1):
            # Produce current from neural network
            i = ind(np.array([x[n][0], x[n][1], x[n][2], x[n][3], x[n][4], x[n][5], Zh, Zh_dt]))
            # Compute future state of the system with differential equation
            ode = ODE(x[n][0], x[n][1], x[n][2], x[n][3], Zh, Zh_dt, i, dt)
            # Set the next state
            x[n+1] = [ode.Zb, ode.Zt, ode.Zb_dt, ode.Zt_dt, ode.Zb_dtdt, ode.Zt_dtdt]
            # Get the next profile
            Zh = profile[n+1]
            Zh_dt = (Zh - profile[n]) / dt
        return x

    x = solver_dampingForce(initialSuspensionState, profile)

    Zb = x[:, 0]
    Zt = x[:, 1]
    Zb_dtdt = x[:, 4]
    Zt_dtdt = x[:, 5]

    def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    # compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = butter_bandpass(0.4, 3, int(1 / dt), 2)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)

    # calculate variance alpha_1
    varZb_dtdt = np.var(z)

    # compute bandpass 2nd order from 0.4 - 3 Hz
    b, a = butter_bandpass(10, 30, int(1 / dt), 1)
    zi = signal.lfilter_zi(b, a)
    z1, _ = signal.lfilter(b, a, Zb_dtdt, zi=zi)
    # calculate variance alpha_2
    varZb_dtdt_h = np.var(z1)

    # compute T_target
    target = K * varZb_dtdt_h + varZb_dtdt

    # check bounding condition

    # do not change constants parameter
    Mb = 500  # mass quarter body [kg]
    Mt = 50  # mass tire + suspension system [kg]
    # do not change constants parameter

    # standard deviation of Zt_dtdt
    devZt_dtdt = np.std(Zt_dtdt * Mt)

    # boundary condition
    F_stat_bound = (Mb + Mt) * 9.81 / 3.0
    if devZt_dtdt > (F_stat_bound):
        return None
    else:
        return target
