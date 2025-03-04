ean_trajectory, std_trajectory = compute_mean_std(truncated_trajectories)

    # # 3) Save the mean and std trajectories if desired
    # np.savetxt("Capstone/Filter/expert_data/mean_trajectory.txt", mean_trajectory, fmt="%.6f")
    # np.savetxt("Capstone/Filter/expert_data/std_trajectory.txt", std_trajectory, fmt="%.6f")

    # print("Mean trajectory shape:", mean_trajectory.shape)
    # print("STD trajectory shape:", std_trajectory.shape)

    # # 4) Plot the mean trajectory (x, y, z) in 3D using matplotlib
    # x = mean_trajectory[:, 0]
    # y = mean_trajectory[:, 1]
    # z = mean_trajectory[:, 2]

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')  # Create a 3D axes
    # ax.plot(x, y, z, marker='o')           # Plot the 3D line
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Mean Trajectory')
    # plt.show()