import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import glob

# Define standard deviation threshold for acceptable range
STD_ACCEPTABLE = 1

def load_trajectories(file_list):
    """Loads all trajectory files into a list of NumPy arrays."""
    return [np.loadtxt(fname) for fname in file_list]

def compute_statistics_spatial(file_list):
    """
    Computes mean and standard deviation based on nearest spatial neighbors.
    Uses filtered_data_1.txt as the reference trajectory.
    """
    trajectories = load_trajectories(file_list)
    
    # Use the first file as reference
    ref_traj = trajectories[0]  # filtered_data_1.txt

    spatial_means = []
    spatial_stds = []

    for ref_idx, ref_point in enumerate(ref_traj):
        matched_points = []  # Store closest points from all other files

        for traj in trajectories[1:]:  # Skip reference trajectory
            if len(traj) == 0:
                continue  # Skip empty trajectories
            
            # Query traj to find which of its points is closest to the current ref_point
            _, nearest_idx = cKDTree(traj[:, :3]).query(ref_point[:3], k=1)  

            matched_points.append(traj[nearest_idx])  # Store nearest traj point to ref_point

        matched_points = np.vstack(matched_points)  # Convert to NumPy array
        
        # Compute mean and standard deviation at this reference point
        mean_point = np.mean(matched_points, axis=0)
        std_point = np.std(matched_points, axis=0)

        spatial_means.append(mean_point)
        spatial_stds.append(std_point)

    # Convert to NumPy arrays
    mean_traj = np.array(spatial_means)
    std_traj = np.array(spatial_stds)

    # Scale standard deviation bounds using STD_ACCEPTABLE
    upper_bound = mean_traj + (STD_ACCEPTABLE * std_traj)
    lower_bound = mean_traj - (STD_ACCEPTABLE * std_traj)

    np.savetxt("Capstone/SignalProcessing/expert_data/left-vein/middle/mean_traj.txt", mean_traj, fmt="%.6f")

    return mean_traj, upper_bound, lower_bound

def get_mean_std_bounds():
    """Retrieve mean, upper, and lower bound trajectories based on spatial proximities."""
    folder_path = "Capstone/SignalProcessing/expert_data/left-vein/middle/"
    file_list = sorted(glob.glob(f"{folder_path}filtered_data_*.txt"))
    return compute_statistics_spatial(file_list), load_trajectories(file_list)

def plot_trajectories(mean_traj, upper_bound, lower_bound, trajectories):
    """Plots the mean trajectory along with scaled upper and lower standard deviation bounds."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Extract x, y, z components
    x_mean, y_mean, z_mean = mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2]
    x_upper, y_upper, z_upper = upper_bound[:, 0], upper_bound[:, 1], upper_bound[:, 2]
    x_lower, y_lower, z_lower = lower_bound[:, 0], lower_bound[:, 1], lower_bound[:, 2]

    # Plot mean trajectory
    ax.plot(x_mean, y_mean, z_mean, 'k-', linewidth=2, label="Mean Trajectory")

    # Plot scaled standard deviation upper/lower bounds as lines
    ax.plot(x_upper, y_upper, z_upper, 'r--', linewidth=1, label=f"Upper Bound (+{STD_ACCEPTABLE}σ)")
    ax.plot(x_lower, y_lower, z_lower, 'b--', linewidth=1, label=f"Lower Bound (-{STD_ACCEPTABLE}σ)")

    # Load live trajectory (extract the first trajectory since it's a single file)
    live_trajectory = load_trajectories(["Capstone/Filter/filtered_data.txt"])[0]

    # Remove NaN values (just in case)
    live_trajectory = np.nan_to_num(live_trajectory)

    # Plot live trajectory in green
    ax.plot(live_trajectory[:, 0], live_trajectory[:, 1], live_trajectory[:, 2], 'g-', linewidth=2, label="LIVE")

    # Plot all individual trajectories
    # for i, traj in enumerate(trajectories):
    #     ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.4, label=f'Trajectory {i+1}')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Trajectory Mean with {STD_ACCEPTABLE}σ Standard Deviation Bounds')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    (mean_traj, upper_bound, lower_bound), trajectories = get_mean_std_bounds()
    plot_trajectories(mean_traj, upper_bound, lower_bound, trajectories)
