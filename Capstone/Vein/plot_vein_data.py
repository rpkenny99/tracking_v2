import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """Loads Tx, Ty, Tz values from a text file."""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1], data[:, 2]

# Load data from text files
Tx_left, Ty_left, Tz_left = load_data('Capstone/Vein/left_vein.txt')
Tx_right, Ty_right, Tz_right = load_data('Capstone/Vein/right_vein.txt')

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(Tx_left, Ty_left, Tz_left, 'r-', label='Left Vein')
ax.plot(Tx_right, Ty_right, Tz_right, 'b-', label='Right Vein')

# Labels and title
ax.set_xlabel('Tx')
ax.set_ylabel('Ty')
ax.set_zlabel('Tz')
ax.set_title('3D Plot of Left and Right Vein Data')
ax.legend()

plt.show()
