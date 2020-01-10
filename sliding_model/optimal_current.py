
from auxiliary import *

def calculate_optimal_I(window_path):
    """
    Given the path to a pickle containing a dictonary with windows,
    this function generates the best possible current, by buidling a compositon
    best guesses for each window. Each window current is brute forced.
    """
    wl, w_nr, fn, windows, s_length, s_factor = load_pickle(window_path)

if __name__=="__main__":
    calculate_optimal_I("")
