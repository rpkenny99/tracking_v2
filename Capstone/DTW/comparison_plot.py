import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

def load_trajectories(file_list):
    """Loads all trajectory files into a list of NumPy arrays."""
    return [np.loadtxt(fname) for fname in file_list]

def plot_trajectories(live_trajectory, expert_trajectory):
    """Plots the live trajectory and expert trajectory in a single 3D graph."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Extract x, y, z components for live and expert trajectories
    x_live, y_live, z_live = live_trajectory[:, 0], live_trajectory[:, 1], live_trajectory[:, 2]
    x_expert, y_expert, z_expert = expert_trajectory[:, 0], expert_trajectory[:, 1], expert_trajectory[:, 2]

    # Plot live trajectory in green
    ax.plot(x_live, y_live, z_live, 'g-', linewidth=2, label="Live Trajectory")

    # Plot expert trajectory in blue
    ax.plot(x_expert, y_expert, z_expert, 'b--', linewidth=2, label="Expert Trajectory")

    # Highlight start and end points for live trajectory
    ax.scatter(x_live[0], y_live[0], z_live[0], color='green', marker='o', s=100, label='Live Start')
    ax.scatter(x_live[-1], y_live[-1], z_live[-1], color='green', marker='x', s=100, label='Live End')

    # Highlight start and end points for expert trajectory
    ax.scatter(x_expert[0], y_expert[0], z_expert[0], color='blue', marker='o', s=100, label='Expert Start')
    ax.scatter(x_expert[-1], y_expert[-1], z_expert[-1], color='blue', marker='x', s=100, label='Expert End')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Live vs Expert Trajectory Comparison')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Load live trajectory
    live_trajectory = load_trajectories(["Capstone/Filter/filtered_data.txt"])[0]

    # Load expert trajectory
    expert_trajectory = load_trajectories(["Capstone/SignalProcessing/expert_data/right-vein/middle/filtered_data_1.txt"])[0]

    # Plot the trajectories
    plot_trajectories(live_trajectory, expert_trajectory)
