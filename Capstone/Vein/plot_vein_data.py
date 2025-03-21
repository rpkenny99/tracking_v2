import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    """Loads Tx, Ty, Tz values from a text file."""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1], data[:, 2]

# Load data from text files
Tx_left, Ty_left, Tz_left = load_data(r'Capstone/SignalProcessing/expert_data/middle/left_vein.txt')
Tx_right, Ty_right, Tz_right = load_data(r'Capstone/SignalProcessing/expert_data/middle/right_vein.txt')
Tx_path, Ty_path, Tz_path = load_data(r'Capstone/SignalProcessing/expert_data/middle/filtered_data_4.txt')

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(Tx_left, Ty_left, Tz_left, 'r-', label='Left Vein')
ax.plot(Tx_right, Ty_right, Tz_right, 'b-', label='Right Vein')
ax.plot(Tx_path, Ty_path, Tz_path, 'b-', label='Path')

# Labels and title
ax.set_xlabel('Tx')
ax.set_ylabel('Ty')
ax.set_zlabel('Tz')
ax.set_title('3D Plot of Left and Right Vein Data')
ax.legend()

plt.show()
