from auxiliary import time_and_profile
import csv

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

for dataset in datasets:
    fname = profileLocation + dataset
    for vel in range(5, 31):  # loop over all velocities
        time, profile = time_and_profile(fname, vel)
        with open(profileLocation + "preproc/" + dataset[:-4] + "_" + str(vel) + ".csv", mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            for p in profile:
                writer.writerow([p])
