import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load vein data from Excel files
left_vein_file = "leftveinvein2.xlsx"  # Replace with the actual file path
right_vein_file = "rightvein2.xlsx"  # Replace with the actual file path

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
angle = np.radians(-90)  # Rotate around the x-axis to flatten
rotation_matrix = np.array([[1, 0, 0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])

points_left_flat = points_left_translated @ rotation_matrix.T
points_right_flat = points_right_translated @ rotation_matrix.T

# Step 5: Plot original and transformed veins
fig = plt.figure(figsize=(10, 5))

# Original veins
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(points_left[:, 0], points_left[:, 1], points_left[:, 2], 'b', label="Left Vein")
ax1.plot(points_right[:, 0], points_right[:, 1], points_right[:, 2], 'r', label="Right Vein")
ax1.set_title("Original Veins")
ax1.set_xlabel("Tx")
ax1.set_ylabel("Ty")
ax1.set_zlabel("Tz")
ax1.legend()

# Transformed veins
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(points_left_flat[:, 0], points_left_flat[:, 1], points_left_flat[:, 2], 'b', label="Left Vein (Flat)")
ax2.plot(points_right_flat[:, 0], points_right_flat[:, 1], points_right_flat[:, 2], 'r', label="Right Vein (Flat)")
ax2.set_title("Transformed Veins (Flat)")
ax2.set_xlabel("Tx")
ax2.set_ylabel("Ty")
ax2.set_zlabel("Tz")
ax2.legend()

plt.show()
