from DTW.DTW_working import *
from multiprocessing import Lock

lock = Lock()
fp = r"Capstone/SignalProcessing/expert_data/left-vein/middle"

# Call Dynamic Time Warping
compute_dtw(fp, lock)