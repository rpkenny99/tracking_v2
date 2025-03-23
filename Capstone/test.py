from DTW.DTW_working import *

# Call Dynamic Time Warping
trainee_data = load_multidimensional_data_as_dataframe(r"Capstone/Filter/filtered_data.txt")
expert_data = load_multidimensional_data_as_dataframe(r"Capstone/SignalProcessing/expert_data/left-vein/middle/mean_traj.txt")
similarity_scores = compute_dtw_per_dimension_parallel(trainee_data, expert_data)