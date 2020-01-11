import os
import csv
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_df(fn, load_path='/net/projects/scratch/winter/valid_until_31_July_2020/hackathon/datasets'):
    colnames = ['time', 'trip', 'profile']
    return pd.read_csv(load_path + '/' + fn, names=colnames, header=None), float((".").join(fn.split("_")[-1].split(".")[:-1]))

def files(path='/net/projects/scratch/winter/valid_until_31_July_2020/hackathon/datasets', end_string=""):
    """
    Iterator for getting a files that end with supplied end_string. Does not show folders.

    Args:
        path          Path to directory to look for files in
        end_string    String ending of file name. Default is empty --> Show everything
    """
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and file.endswith(end_string):
            yield file

def file_list():
    """
    Return a list of all data files.
    """
    return [f for f in files(end_string=".csv")]

def df_dict(path='/net/projects/scratch/winter/valid_until_31_July_2020/hackathon/datasets'):
    """
    Return a dictionary, filled with all dataframes.
    """
    colnames = ['time', 'trip', 'profile']
    data_frames = {}
    for indx_f, f in enumerate(file_list()):
        df = pd.read_csv(path + '/' + f, names=colnames, header=None)
        data_frames[indx_f] = (f, df)

    return data_frames


def transform_velocity(df, vel, dt=0.005):
    """
    Transforms dataframe with velocity

    Parameters:
    file_name (str): filename including path (CSV)
    vel (int): constant velocity
    path (str): path to where data is stored
    dt (float): dt step size

    Returns:
    new_df: data frame with fitted dimensions.
    """
    colnames = ['time', 'trip', 'profile']

    # get simulation time by constant speed
    T = float(list(df['trip'])[-1]) / float(vel)


    # get number of dt - timesteps
    N = int(np.round(T / dt))

    # array for time
    time = np.linspace(0, T, N + 1)


    # get driving speed vector e.g for dynamic (non constant) speed
    v = np.ones(time.size) * vel

    # get trip at each dt
    trip = []
    for i in range(0, time.size):
        trip.append(np.trapz(v[0:i + 1], dx=dt))

    # get the road profile by the tripRecording
    profile = np.interp(trip, df['trip'], df['profile'])

    # create new df using new values
    new_df = pd.DataFrame(list(zip(time, trip, profile)), columns=colnames)

    return new_df

def save_as_pickle(dictionary, file_name, path='/net/projects/scratch/winter/valid_until_31_July_2020/dcelik/windows'):
    """
    Save a dictionary as a pickle file.

    Args:
        dictionary             dicationary to be pickled
        file_name              name of the pickled file (no extension)
        path                   path of where the file is to be stored
    """
    with open(path + '/' + file_name + '.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("[+] saved dictionary in {}!".format(path + '/' + file_name + '.pickle'))


def load_pickle(file_path):
    """
    Loads a pickle file.

    Args:
        file_path              path to file to be depickled

    Returns:
        window length
        window number
        file name
        list of windows
        sliding length (only if sliding was used, else None)
        sliding factor (only if sliding was used, else None)
    """
    with open(file_path, 'rb') as handle:
        df = pickle.load(handle)
    print("[+] loaded dictionary from {}!".format(file_path))

    if 'slide_factor' in df.keys():
        return_tuple = (df['windows_length'], df['windows_nr'], df['file_name'],
                df['data_frames'], df['sliding_length'], df['slide_factor'])
    else:
        return_tuple = (df['windows_length'], df['windows_nr'], df['file_name'],
                df['data_frames'], None, None)

    return return_tuple

if __name__=="__main__":
    f_list = file_list()
    f = f_list[0]
    df = df_time_and_profile(f, vel=15)
    df.plot(x="time", y="profile")
    df.plot(x="trip", y="profile")
    plt.show()
