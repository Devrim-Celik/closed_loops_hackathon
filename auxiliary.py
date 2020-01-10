import numpy as np
import csv

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
