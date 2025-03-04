import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load vein data from Excel files
left_vein_file = "Capstone/Vein/leftveinvein2_smoothed4.xlsx"  # Replace with the actual file path
right_vein_file = "Capstone/Vein/rightvein2.xlsx"  # Replace with the actual file path

# Assuming Excel files have columns Tx, Ty, Tz for vein coordinates
left_data = pd.read_excel(left_vein_file)
right_data = pd.read_excel(right_vein_file)

# Extract coordinates for left and right veins
Tx_left, Ty_left, Tz_left = left_data['Tx'].to_numpy(), left_data['Ty'].to_numpy(), left_data['Tz'].to_numpy()
Tx_right, Ty_right, Tz_right = right_data['Tx'].to_numpy(), right_data['Ty'].to_numpy(), right_data['Tz'].to_numpy()

# Step 2: Filter out invalid entries (e.g., extremely low values)
threshold = -1e10  # Define threshold for invalid values

# Apply filter for left vein
valid_indices_left = (Tx_left > threshold) & (Ty_left > threshold) & (Tz_left > threshold)
Tx_left, Ty_left, Tz_left = Tx_left[valid_indices_left], Ty_left[valid_indices_left], Tz_left[valid_indices_left]

# Apply filter for right vein
valid_indices_right = (Tx_right > threshold) & (Ty_right > threshold) & (Tz_right > threshold)
Tx_right, Ty_right, Tz_right = Tx_right[valid_indices_right], Ty_right[valid_indices_right], Tz_right[valid_indices_right]

# Combine points for easier processing
points_left = np.vstack((Tx_left, Ty_left, Tz_left)).T
points_right = np.vstack((Tx_right, Ty_right, Tz_right)).T

# Step 3: Translation - move veins to the desired plane (e.g., z = 0)
translation_vector = np.array([0, 0, -np.mean(Tz_left)])  # Adjust Z to align with the glass
points_left_translated = points_left + translation_vector
points_right_translated = points_right + translation_vector

# Step 4: Rotation - rotate veins to lie flat (e.g., onto the x-y plane)
# Define a rotation matrix for aligning with the x-y plane
angle_x = np.radians(-90)  # Rotate around the x-axis to flatten
rotation_matrix_x = np.array([[1, 0, 0],
                               [0, np.cos(angle_x), -np.sin(angle_x)],
                               [0, np.sin(angle_x), np.cos(angle_x)]])

points_left_flat = points_left_translated @ rotation_matrix_x.T
points_right_flat = points_right_translated @ rotation_matrix_x.T

# Step 5: Rotate points 90 degrees about the y-axis
angle_y = np.radians(90)  # Rotate around the y-axis
rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                               [0, 1, 0],
                               [-np.sin(angle_y), 0, np.cos(angle_y)]])

points_left_rotated = points_left_flat @ rotation_matrix_y.T
points_right_rotated = points_right_flat @ rotation_matrix_y.T

# Step 6: Translate all points such that the last point of the left vein is at the origin
final_translation_vector = -points_left_rotated[-1]
points_left_rotated = points_left_rotated + final_translation_vector
points_right_rotated = points_right_rotated + final_translation_vector

# Save transformed veins to separate text files
np.savetxt("left_veins.txt", points_left_rotated, fmt="%.6f")
np.savetxt("right_veins.txt", points_right_rotated, fmt="%.6f")

# Step 7: Plot original and transformed veins
fig = plt.figure(figsize=(10, 5))

def set_axes_equal(ax):
    """Set 3D plot axes to the same scale."""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    max_range = np.ptp(limits)
    mean_vals = np.mean(limits, axis=1)
    ax.set_xlim(mean_vals[0] - max_range/2, mean_vals[0] + max_range/2)
    ax.set_ylim(mean_vals[1] - max_range/2, mean_vals[1] + max_range/2)
    ax.set_zlim(mean_vals[2] - max_range/2, mean_vals[2] + max_range/2)

ax = fig.add_subplot(111, projection='3d')
ax.plot(points_left_rotated[:, 0], points_left_rotated[:, 1], points_left_rotated[:, 2], 'b-', label="Left Vein (Aligned)")
ax.plot(points_right_rotated[:, 0], points_right_rotated[:, 1], points_right_rotated[:, 2], 'r-', label="Right Vein (Aligned)")

# Labels and title
ax.set_xlabel("Tx")
ax.set_ylabel("Ty")
ax.set_zlabel("Tz")
ax.set_title("Veins Aligned to Last Left Vein Point")
ax.legend()

plt.show()
