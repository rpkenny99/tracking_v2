import numpy as np
import matplotlib.pyplot as plt

def load_vein_data(filename):
    """Load vein data from a text file."""
    return np.loadtxt(filename)

def translate_points(points, translation_vector):
    """Translate points by a given translation vector."""
    return points + translation_vector

def rotate_points_z(points, angle_deg):
    """Rotate points around the Z-axis by a given angle in degrees."""
    angle_rad = np.radians(angle_deg)
    rotation_matrix_z = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    return points @ rotation_matrix_z.T

# Load original vein data
left_vein = load_vein_data("left_veins.txt")
right_vein = load_vein_data("right_veins.txt")

# Step 1: Rotate veins 90 degrees about the Z-axis
left_vein_rotated = rotate_points_z(left_vein, 270)
right_vein_rotated = rotate_points_z(right_vein, 270)

# Step 2: Realign the last point of left-vein to the origin
origin_shift = left_vein_rotated[-1]
left_vein_aligned = left_vein_rotated - origin_shift
right_vein_aligned = right_vein_rotated - origin_shift

# Step 3: Apply final translation adjustments
# y_adjust, x_adjust, z_adjust = -105.60215667741426, -31.6512872437187, 61.668481691610054
x_adjust, y_adjust, z_adjust = -84.46442979437285, -30.568956342599517, 53.507186202558685
translation_vector = np.array([x_adjust, y_adjust, z_adjust])

left_vein_final = translate_points(left_vein_aligned, translation_vector)
right_vein_final = translate_points(right_vein_aligned, translation_vector)

# Save final veins to new files
np.savetxt("left_vein_final.txt", left_vein_final, fmt="%.6f")
np.savetxt("right_vein_final.txt", right_vein_final, fmt="%.6f")

# Plot final veins
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(left_vein_final[:, 0], left_vein_final[:, 1], left_vein_final[:, 2], 'g-', label="Left Vein Final")
ax.plot(right_vein_final[:, 0], right_vein_final[:, 1], right_vein_final[:, 2], 'm-', label="Right Vein Final")

ax.set_xlabel("Tx")
ax.set_ylabel("Ty")
ax.set_zlabel("Tz")
ax.set_title("Final Rotated and Translated Veins")
ax.legend()
plt.show()
