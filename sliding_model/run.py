import matplotlib as plt
from sliding_window import generate_windows
from auxiliary import *

def run_sliding_window_model(file_name, velocity, dt=0.005, I_levels=9,
    save_path="/net/projects/scratch/winter/valid_until_31_July_2020/dcelik"):
    """

    Args
        file_name (str): file_name including the .csv ending
    """
    # get data and K
    df, K = load_df(file_name)
    print("[+] Loaded file {} with K={}!".format(file_name, K))
    # tranform using velocity
    df = transform_velocity(df, vel=velocity, dt=dt)
    # generate I and add it to df
    optimal_I, errors = generate_windows(df, vel=velocity, fn=file_name, K=K, dt=dt,
        I_levels=I_levels)
    df["optimal_I"] = optimal_I
    df.to_csv(save_path + "/OPTIMAL-" + file_name)
    print("[+] Saved finish dataframe with current in {}!".format(save_path + "/OPTIMAL-" + file_name))
    df.plot(x="time", y=["profile", "optimal_I"])
    plt.figure()
    plt.plot(errors)
    plt.show()

    return optimal_I

if __name__=="__main__":
    run_sliding_window_model(file_name="ts1_1_k_3.0.csv", velocity=20, I_levels=3)

"""
if __name__=="__main__":
    print("START")
    list_of_old_df = df_dict()
    print("LOG1")
    print([f for (f, df) in list_of_old_df.values()])
    for i in range(len(list_of_old_df)):
        f, df = list_of_old_df[i]
        if f == "ts2_k_20.0.csv":
            break
    df = df_time_and_profile(df, vel=20)
    print("LOG2")
    #df["profile"].to_csv("test_profiprofil.csv")

    #df = pd.read_csv("test_profiprofil.csv", names=["profile"])
    df["profile"].plot()
    plt.show()
    f = "test_placeholder"
    optimal_I = generate_windows(df, vel=20, fn=f, K=20)
    df["optimal_I"] = optimal_I/100
    df["idx"] = list(range(len(df)))
    df.to_csv("with_optimal.csv")

    df.plot(x="idx", y=["profile"])
    df.plot(x="idx", y=["profile", "optimal_I"])
    plt.show()
"""
