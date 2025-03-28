import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_trajectories(file_list):
    """Loads all trajectory files into a list of NumPy arrays."""
    return [np.loadtxt(fname) for fname in file_list]

def plot_trajectories(live_trajectory, expert_trajectory):
    """Plots the live and expert (mean) trajectories in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # Extract x, y, z components
    x_live, y_live, z_live = live_trajectory[:, 0], live_trajectory[:, 1], live_trajectory[:, 2]
    x_expert, y_expert, z_expert = expert_trajectory[:, 0], expert_trajectory[:, 1], expert_trajectory[:, 2]

    # Plot trajectories
    ax.plot(x_expert, y_expert, z_expert, 'k-', linewidth=2, label="Expert (Mean) Trajectory")
    ax.plot(x_live, y_live, z_live, 'g-', linewidth=2, label="Live Trajectory")

    # Start/End markers
    ax.scatter(x_live[0], y_live[0], z_live[0], color='green', marker='o', s=100, label='Live Start')
    ax.scatter(x_live[-1], y_live[-1], z_live[-1], color='green', marker='x', s=100, label='Live End')
    ax.scatter(x_expert[0], y_expert[0], z_expert[0], color='black', marker='o', s=100, label='Expert Start')
    ax.scatter(x_expert[-1], y_expert[-1], z_expert[-1], color='black', marker='x', s=100, label='Expert End')

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Live vs Expert Trajectory')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load trajectories
    live_trajectory = load_trajectories(["Capstone/Filter/filtered_data.txt"])[0]
    expert_trajectory = load_trajectories(["Capstone/SignalProcessing/expert_data/left-vein/middle/mean_traj.txt"])[0]

    # Plot
    plot_trajectories(live_trajectory, expert_trajectory)
