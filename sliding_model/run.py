from sliding_window import generate_windows
from auxiliary import *


def p(txt):
    print("\n\n\n\n\n\n" + txt + "\n\n\n\n\n\n")

if __name__=="__main__":
    p("START")
    list_of_old_df = df_dict()
    p("LOG1")
    f, df = list_of_old_df[0]
    df = df_time_and_profile(df, vel=10)
    p("LOG2")
    best_I = generate_windows(df, vel=10, fn=f)
    p("LOG3")

    df.loc[2000: 2020].plot(x="trip", y="profile")
    df.plot(x="trip", y="profile")
    plt.figure()
    plt.plot(best_I)
    plt.show()
