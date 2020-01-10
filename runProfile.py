import csv
from ODE import ODE
import numpy as np
from scipy import signal
from auxiliary import time_and_profile, new_population

# record location
# roadProfileLocation = '/net/projects/scratch/winter/valid_until_31_July_2020/hackathon/datasets/'
profileLocation = 'datasets/'
# profile filenames
datasets = [
    'ts1_1_k_3.0.csv',
    'ts1_2_k_3.0.csv',
    'ts1_3_k_3.0.csv',
    'ts1_4_k_3.0.csv'
    'ts2_k_20.0.csv',
    'ts3_1_k_3.0.csv',
    'ts3_2_k_3.0.csv',
    'ts3_3_k_3.0.csv',
]

# driving speed [m/s] between 2 and 40
vel = 27.0

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

def time_and_profile(fname, vel):
    """Returns the right time and right interpolated profile given a velocity

        Parameters:
        fname (str): filename including path (CSV)
        vel (int): constant velocity

        Returns:
        time array: right time array
        profile array: interpolated Zh (profile)

       """
    timeRecording = []
    tripRecording = []
    ZH = []
    # simulation interval in seconds
    dt = 0.005

    with open(fname) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            timeRecording.append(float(row[0]))
            tripRecording.append(float(row[1]))
            ZH.append(float(row[2]))

    # get simulation time by constant speed
    T = float(tripRecording[-1]) / float(vel)

    N = int(np.round(T / dt))
    time = np.linspace(0, T, N + 1)

    # get driving speed vector e.g for dynamic (non constant) speed
    v = np.ones(time.size) * vel

    # get trip at each dt
    trip = []
    for i in range(0, time.size):
        trip.append(np.trapz(v[0:i + 1], dx=dt))

    # get the road profile by the tripRecording
    profile = np.interp(trip, tripRecording, ZH)

    return time, profile

def new_population():
    pass

def calc_fitness(ind):
    '''
    ind: individual from a population (GA)
    return: target
    '''

    def solver_dampingForce(intialSuspensionState, profile, t):
        '''
        intialSuspensionState: [Zb, Zt, Zb_dt, Zt_dt] at t=0
        H: road profile [m] over time intervals
        t: time intervals
        '''
        N = t.size - 1  # No of time intervals
        x = np.zeros((N + 1, 6))  # array of [Zb, Zt, Zb_dt, Zt_dt, Zb_dtdt, Zt_dtdt] at each time interval
        x[0] = intialSuspensionState  # [Zb, Zt, Zb_dt, Zt_dt, Zb_dtdt, Zt_dtdt] at t=0
        for n in range(0, N):
            Zh = profile[n]
            dt = float(t[1] - t[0])  # dt
            Zh_dt = float(0)
            if n > 0:
                Zh_prev = profile[n - 1]
                dt = t[n] - t[n - 1]
                Zh_dt = (Zh - Zh_prev) / dt
            i = ind(np.array([x[n][0], x[n][1], x[n][2], x[n][3], x[n][4], x[n][5], Zh, Zh_dt]))
            ode = ODE(x[n][0], x[n][1], x[n][2], x[n][3], Zh, Zh_dt, i, dt)
            x[n + 1] = [ode.Zb, ode.Zt, ode.Zb_dt, ode.Zt_dt, ode.Zb_dtdt, ode.Zt_dtdt]
        return x

    x = solver_dampingForce(initialSuspensionState, ZH, time)

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


avg_fitnesses = np.array()
for dataset in datasets:
    # road specific weight factor
    tmp = dataset[0].split("_k_")
    tmp = tmp[1]
    K = float(tmp[:-4])
    time, ZH = time_and_profile(profileLocation + dataset, vel)
    pop = new_population()
    fitnesses = []
    for individual in pop:
        fitnesses.append(calc_fitness(individual))
    fitnesses = np.array(fitnesses)
    avg_fitnesses += fitnesses
avg_fitnesses /= len(datasets)
