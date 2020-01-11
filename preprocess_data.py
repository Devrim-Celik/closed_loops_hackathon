from auxiliary import time_and_profile
import csv
import time as time_module

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
    print('\n%% Processing dataset', dataset)
    fname = profileLocation + dataset
    start = time_module.time()
    for vel in range(5, 31):  # loop over all velocities
        print('… velocity', vel)
        time, profile = time_and_profile(fname, vel)
        with open(profileLocation + "preproc/" + dataset[:-4] + "_vel_" + str(vel) + ".csv", mode='w') as file:
            writer = csv.writer(file, delimiter=',')
            for p in profile:
                writer.writerow([p])
    print(f'Took {time_module.time()-start:.3f} seconds to process dataset.')
