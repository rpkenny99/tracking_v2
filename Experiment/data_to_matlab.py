import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# File path
file_path = 'data.csv'

# Load the data
data = pd.read_csv(file_path)

# Extract Tx, Ty, Tz columns
tx = data['Center_X']
ty = data['Center_Y']
tz = data['Center_Z']

# Plot the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tx, ty, tz, c='r', marker='o')

# Label axes
ax.set_xlabel('Center_X')
ax.set_ylabel('Center_Y')
ax.set_zlabel('Center_Z')

# Show plot
plt.show()
