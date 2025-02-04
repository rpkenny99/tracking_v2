import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# File path
file_path = 'Experiment/3_pattern.csv'

# Load the data
data = pd.read_csv(file_path)

# Extract Tx, Ty, Tz columns
tx = data['Tx']
ty = data['Ty']
tz = data['Tz']

# Plot the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tx, ty, tz, c='r', marker='o')

# Label axes
ax.set_xlabel('Tx')
ax.set_ylabel('Ty')
ax.set_zlabel('Tz')

# Show plot
plt.show()
