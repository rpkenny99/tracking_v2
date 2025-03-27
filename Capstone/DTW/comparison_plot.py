# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def load_trajectories(file_list):
#     """Loads all trajectory files into a list of NumPy arrays."""
#     return [np.loadtxt(fname) for fname in file_list]

# def plot_trajectories(live_trajectory, expert_trajectory, most_different_dims=None):
#     """Plots trajectories and highlights dimensions with high DTW dissimilarity."""
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(projection='3d')

#     # Extract x, y, z components
#     x_live, y_live, z_live = live_trajectory[:, 0], live_trajectory[:, 1], live_trajectory[:, 2]
#     x_expert, y_expert, z_expert = expert_trajectory[:, 0], expert_trajectory[:, 1], expert_trajectory[:, 2]

#     # Default: Plot all trajectories normally
#     ax.plot(x_live, y_live, z_live, 'g-', linewidth=2, label="Live Trajectory", alpha=0.5)
#     ax.plot(x_expert, y_expert, z_expert, 'b--', linewidth=2, label="Expert Trajectory", alpha=0.5)

#     # # Highlight most dissimilar dimensions (if provided)
#     # if most_different_dims is not None:
#     #     dim_labels = ['X', 'Y', 'Z']
#     #     highlight_colors = ['r', 'm', 'c']  # Red, Magenta, Cyan for each problematic dimension
        
#     for i, dim in enumerate(most_different_dims):
#         if dim < 3:  # Ensure we only handle X/Y/Z (assuming 3D data)
#             # Highlight LIVE trajectory (thicker + colored)
#             if dim == 0:
#                 ax.plot(x_live, y_live, z_live, 
#                         color=highlight_colors[i], linewidth=4, alpha=0.7, 
#                         label=f'Live Dim {dim_labels[dim]} (High DTW)')
#             elif dim == 1:
#                 ax.plot(x_live, y_live, z_live, 
#                         color=highlight_colors[i], linewidth=4, alpha=0.7)
#             elif dim == 2:
#                 ax.plot(x_live, y_live, z_live, 
#                         color=highlight_colors[i], linewidth=4, alpha=0.7)
            
#             # Add annotation for the dimension
#             ax.text(x_live.mean(), y_live.mean(), z_live.mean(), 
#                     f'Dim {dim_labels[dim]} Diff', color=highlight_colors[i],
#                     fontsize=10, weight='bold')

#     # Start/End markers
#     ax.scatter(x_live[0], y_live[0], z_live[0], color='green', marker='o', s=100, label='Live Start')
#     ax.scatter(x_live[-1], y_live[-1], z_live[-1], color='green', marker='x', s=100, label='Live End')
#     ax.scatter(x_expert[0], y_expert[0], z_expert[0], color='blue', marker='o', s=100, label='Expert Start')
#     ax.scatter(x_expert[-1], y_expert[-1], z_expert[-1], color='blue', marker='x', s=100, label='Expert End')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Trajectory Comparison (Red = Most Dissimilar Dimensions)')
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     # Load trajectories
#     live_trajectory = load_trajectories(["Capstone/Filter/filtered_data.txt"])[0]
#     expert_trajectory = load_trajectories(["Capstone/SignalProcessing/expert_data/right-vein/middle/mean_traj.txt"])[0]

#     # Plot with highlights
#     plot_trajectories(live_trajectory, expert_trajectory)


def compute_dtw(expert_data="Capstone/SignalProcessing/expert_data/right-vein/middle/mean_traj.txt", 
                file_lock=None, 
                trainee_data="Capstone/Filter/filtered_data.txt"):
    # Load the trainee data
    with file_lock:
        input_data_df = load_multidimensional_data_as_dataframe(trainee_data)

    # Load the single reference expert data
    reference_data_df = load_multidimensional_data_as_dataframe(expert_data)
    reference_name = os.path.basename(expert_data)

    average_similarity = []

    if input_data_df is not None and reference_data_df is not None:
        array1 = input_data_df.to_numpy()
        array2 = reference_data_df.to_numpy()

        similarity_scores = compute_dtw_per_dimension_parallel(array1, array2)

        if np.all(np.isnan(similarity_scores)):
            print(
                f"\nThe similarity scores for {trainee_data} and {reference_name} are all NaN. Skipping analysis.")
        else:
            average_similarity.append(np.nanmean(similarity_scores))

            print(f"\nComparison with {reference_name}:")
            print("Similarity scores for each dimension:", similarity_scores)
            print("Average similarity score across all dimensions:", average_similarity)

            most_different_indices = np.argsort(similarity_scores)[-3:][::-1]
            print(f"The three dimensions with the most difference (highest similarity scores) are: {most_different_indices}")
    else:
        print(f"Failed to load either trainee or expert dataset.")

    return average_similarity
