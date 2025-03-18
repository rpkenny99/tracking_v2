import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy, pandas, matplotlib 

def _plotVeins(self, layout):
        """Plot the vein visualization using matplotlib and overlay it on the black area."""
        # Create a matplotlib figure with a fully transparent background
        self.figure = Figure(facecolor='none')  # Transparent figure
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: transparent;")  # Transparent canvas

        # Set a fixed size for the canvas to make the plot smaller
        self.canvas.setFixedSize(400, 300)  # Adjust the size as needed

        # Add the canvas to the layout
        layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        # Plot the veins
        ax = self.figure.add_subplot(111, projection='3d')
        ax.grid(False)  # Remove the grid

        # Ensure the axis background is fully transparent
        ax.patch.set_alpha(0)

        # Hide axes and labels completely
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)  # Hide spines completely

        # Make all panes fully transparent
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        # Make axis lines fully transparent
        ax.xaxis.line.set_alpha(0)
        ax.yaxis.line.set_alpha(0)
        ax.zaxis.line.set_alpha(0)

        # Load and plot left vein
        left_vein_file = "leftveinvein2_smoothed4.xlsx"
        left_data = pd.read_excel(left_vein_file)
        Tx_left, Ty_left, Tz_left = left_data['Tx'].to_numpy(), left_data['Ty'].to_numpy(), left_data['Tz'].to_numpy()
        threshold = -1e10
        valid_indices_left = (Tx_left > threshold) & (Ty_left > threshold) & (Tz_left > threshold)
        Tx_left, Ty_left, Tz_left = Tx_left[valid_indices_left], Ty_left[valid_indices_left], Tz_left[valid_indices_left]

        # Scale the Tx, Ty, and Tz values by 2 to stretch the veins along all axes
        Tx_left = Tx_left * 2
        Ty_left = Ty_left * 2
        Tz_left = Tz_left * 2

        points_left = np.vstack((Tx_left, Ty_left, Tz_left)).T
        ax.plot(points_left[:, 0], points_left[:, 1], points_left[:, 2], 'b', label="Left Vein")

        # Load and plot right vein
        right_vein_file = "rightvein2.xlsx"
        right_data = pd.read_excel(right_vein_file)
        Tx_right, Ty_right, Tz_right = right_data['Tx'].to_numpy(), right_data['Ty'].to_numpy(), right_data['Tz'].to_numpy()
        valid_indices_right = (Tx_right > threshold) & (Ty_right > threshold) & (Tz_right > threshold)
        Tx_right, Ty_right, Tz_right = Tx_right[valid_indices_right], Ty_right[valid_indices_right], Tz_right[valid_indices_right]

        # Scale the Tx, Ty, and Tz values by 2 to stretch the veins along all axes
        Tx_right = Tx_right * 2
        Ty_right = Ty_right * 2
        Tz_right = Tz_right * 2

        points_right = np.vstack((Tx_right, Ty_right, Tz_right)).T
        ax.plot(points_right[:, 0], points_right[:, 1], points_right[:, 2], 'r', label="Right Vein")

        # Remove title and legend
        ax.set_title("")
        ax.legend().set_visible(False)

        # Ensure the entire figure and axes background are transparent
        ax.set_facecolor('none')
        self.figure.patch.set_alpha(0)  # Make entire figure transparent

        self.canvas.draw()""

       


def main():
     _plotVeins(self, layout):


if __name__ == "__main__":
    main()
