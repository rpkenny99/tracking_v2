import numpy as np
import matplotlib.pyplot as plt

def load_trajectories_no_truncation(file_list):
    """
    Loads each file and returns a list of arrays, one per file.
    No truncation is performed. Each array may have a different number of rows.
    """
    all_data = []
    for fname in file_list:
        data = np.loadtxt(fname)
        all_data.append(data)
    return all_data

if __name__ == "__main__":
    file_list = [
        "Capstone/Filter/expert_data/data_1.txt",
        "Capstone/Filter/expert_data/data_2.txt",
        "Capstone/Filter/expert_data/data_3.txt",
        "Capstone/Filter/expert_data/data_4.txt",
        "Capstone/Filter/expert_data/data_5.txt",
        "Capstone/Filter/expert_data/data_6.txt"
    ]
    
    # 1) Load each file, no truncation
    all_trajectories = load_trajectories_no_truncation(file_list)
    
    # 2) Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # 3) Plot each trajectory's (x, y, z) in the same figure
    for i, traj in enumerate(all_trajectories, start=1):
        x = traj[:, 0]  # column 0 is x
        y = traj[:, 1]  # column 1 is y
        z = traj[:, 2]  # column 2 is z
        ax.plot(x, y, z, marker='.', label=f"Trajectory {i}")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('All Trajectories (x, y, z) — No Truncation')
    ax.legend()
    plt.show()
