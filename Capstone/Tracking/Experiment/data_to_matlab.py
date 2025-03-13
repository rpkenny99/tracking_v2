import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# File path
file_path = 'Capstone/SignalProcessing/expert_data/filtered_data_10.txt'
# file_path = 'Capstone/Filter/filtered_data.txt'
# file_path = 'Capstone/Tracking/data.txt'

# Read the text file using pandas.
# Each line of data.txt has x, y, z, pitch, roll, yaw (6 columns),
# separated by whitespace or spaces.
data = pd.read_csv(
    file_path, 
    sep=r"\s+",               # Use regex to split on whitespace
    header=None,              # No header line in the file
    names=["x", "y", "z", "pitch", "roll", "yaw"]  # Assign column names
)

# Extract x, y, z columns
x = data["x"]
y = data["y"]
z = data["z"]

# Create a 3D scatter plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c="r", marker="o")

# Label the axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.title("3D Scatter of (x, y, z)")
plt.show()

pitch = data["pitch"]
roll = data["roll"]
yaw = data["yaw"]

# Create a figure with 3 subplots, sharing the x-axis
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

axes[0].plot(pitch, label='Pitch', color='blue')
axes[0].set_ylabel("Pitch")
axes[0].legend()

axes[1].plot(roll, label='Roll', color='orange')
axes[1].set_ylabel("Roll")
axes[1].legend()

axes[2].plot(yaw, label='Yaw', color='green')
axes[2].set_xlabel("Sample")
axes[2].set_ylabel("Yaw")
axes[2].legend()

plt.suptitle("Pitch, Roll, and Yaw vs. Sample")
plt.tight_layout()
plt.show()