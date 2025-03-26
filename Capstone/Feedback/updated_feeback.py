from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QMovie, QFont, QColor, QPainter
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel, QGraphicsOpacityEffect,
    QLineEdit, QPushButton, QTextEdit, QGroupBox, QHBoxLayout, QDialog, QStackedWidget
)
import sys
from random import uniform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from queue import Queue
from SignalProcessing.compute_avg_std_dev import get_mean_std_bounds, STD_ACCEPTABLE

import matplotlib
# Hide debug messages (only show warnings and above)
matplotlib.set_loglevel("warning")

class IntroScreen(QDialog):
    """Introductory screen to start the simulation."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cyber-Physical Infant IV Simulator")
        self.showMaximized()  # Make the window maximized

        layout = QVBoxLayout()
        label = QLabel("Cyber-Physical Infant IV Simulator")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        font = QFont()
        font.setPointSize(26)
        font.setBold(True)
        label.setFont(font)

        layout.addWidget(label)

        start_button = QPushButton("Start Simulation")
        start_button.clicked.connect(self.accept)  # Close the dialog on click
        layout.addWidget(start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

class PickVeinScreen(QDialog):
    """Choose the Vein to be pierced."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pick Your Vein")
        self.setFixedSize(1365, 680)

        # Main layout
        layout = QVBoxLayout()

        # Header (moved to the top)
        header_label = QLabel("Pick Your Vein")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        header_label.setFont(font)
        layout.addWidget(header_label)

        # Arm Image
        self.arm_image_label = QLabel(self)
        self.pixmap = QPixmap("Capstone/Feedback/redv3in.png")

        # Adjust the image size here by changing the width and height values
        self.image_width = 950  # Set your desired width in pixels
        self.image_height = 450  # Set your desired height in pixels
        self.arm_image_label.setPixmap(self.pixmap.scaled(
            self.image_width, self.image_height, Qt.AspectRatioMode.KeepAspectRatio
        ))
        self.arm_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.arm_image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Buttons for vein selection
        button_layout = QHBoxLayout()

        # Left Vein Button
        left_vein_button = QPushButton("Left Vein")
        left_vein_button.clicked.connect(self.select_left_vein)
        button_layout.addWidget(left_vein_button)

        # Right Vein Button
        right_vein_button = QPushButton("Right Vein")
        right_vein_button.clicked.connect(self.select_right_vein)
        button_layout.addWidget(right_vein_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def select_left_vein(self):
        """Handle selection of the left vein."""
        self.selected_vein = "Left Vein"
        self.accept()

    def select_right_vein(self):
        """Handle selection of the right vein."""
        self.selected_vein = "Right Vein"
        self.accept()

class PickInsertionPointScreen(QDialog):
    def __init__(self, selected_vein):
        super().__init__()
        self.setWindowTitle("Pick Insertion Point")
        self.setFixedSize(1365, 680)

        # Store the selected vein
        self.selected_vein = selected_vein

        layout = QVBoxLayout()

        # Header    
        header_label = QLabel("Pick Insertion Point")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        header_label.setFont(font)
        layout.addWidget(header_label)

        # Arm Image with Clickable Points
        self.arm_image_label = QLabel(self)
        
        # Load the appropriate image based on the selected vein
        if self.selected_vein == "Left Vein":
            self.pixmap = QPixmap("Capstone/Feedback/leftvein-removebg-preview.png")
        elif self.selected_vein == "Right Vein":
            self.pixmap = QPixmap("Capstone/Feedback/rightvein-removebg-preview.png")
        else:
            # Default image if no vein is selected (optional)
            self.pixmap = QPixmap("Capstone/Feedback/default_image.png")

        self.arm_image_label.setPixmap(self.pixmap.scaled(950, 450, Qt.AspectRatioMode.KeepAspectRatio))
        self.arm_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.arm_image_label)

        # Buttons for insertion points
        button_layout = QHBoxLayout()
        point_a_button = QPushButton("Top")
        point_a_button.clicked.connect(self.select_point_a)
        button_layout.addWidget(point_a_button)

        point_b_button = QPushButton("Middle")
        point_b_button.clicked.connect(self.select_point_b)
        button_layout.addWidget(point_b_button)

        point_c_button = QPushButton("Bottom")
        point_c_button.clicked.connect(self.select_point_c)
        button_layout.addWidget(point_c_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def select_point_a(self):
        """Handle selection of Point A."""
        self.selected_point = "Point A"
        self.accept()

    def select_point_b(self):
        """Handle selection of Point B."""
        self.selected_point = "Point B"
        self.accept()

    def select_point_c(self):
        """Handle selection of Point C."""
        self.selected_point = "Point C"
        self.accept()

class FeedbackUI(QMainWindow):
    def __init__(self,
                 selected_vein,
                 selected_point,
                 max_updates=12,
                 update_interval=5,
                 work_queue=None,
                 angle_range_queue=None,
                 app_to_signal_processing=None,
                 direction_intruction_queue = None):
        super().__init__()
        self.setWindowTitle("Feedback UI - Needle Insertion")
        self.showFullScreen()  # Make the window maximized

        self.live_trajectory_queue = Queue()

        # Store the selected vein and insertion point
        self.selected_vein = selected_vein
        self.selected_point = selected_point
        self.work_queue = work_queue
        self.angle_range_queue = angle_range_queue
        self.app_to_signal_processing = app_to_signal_processing
        self.direction_intruction_queue = direction_intruction_queue

        self.expert_pitch, _, self.expert_yaw, self.expert_pitch_std, _, self.expert_yaw_std = angle_range_queue.get()

        # Define initial view parameters for the 3D plot
        self.original_elev = 30  # Default elevation
        self.original_azim = -60  # Default azimuth

        # Debug statements
        print(f"FeedbackUI - Selected Vein: {self.selected_vein}")
        print(f"FeedbackUI - Selected Point: {self.selected_point}")

        # Set up the main layout
        self.generalLayout = QHBoxLayout()
        centralWidget = QWidget(self)
        centralWidget.setLayout(self.generalLayout)
        self.setCentralWidget(centralWidget)

        self.sessionLog = []
        self.update_count = 0
        self.max_updates = max_updates
        self.total_angle_deviation = 0  # To accumulate angle deviation
        self.total_depth_deviation = 0  # To accumulate depth deviation

        # Initialize the pixmap attribute with a default image
        self.pixmap = QPixmap("Capstone/Feedback/default_image.png")  # Ensure this image exists

        self._createDisplay()

        # Timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._updateData)
        self.timer.start(10)

    def _createDisplay(self):
        """Create the UI layout with the updated design."""
        # Session Timer in top-left corner
        self.session_time_label = QLabel("00:00", self)
        self.session_time_label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 150);
                color: white;
                font-size: 32px;
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 5px;
            }
        """)
        self.session_time_label.move(10, 10)
        self.session_time_label.adjustSize()
        self.session_time_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.session_time_label.show()

        # Timer logic
        self.elapsed_seconds = 0
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self._updateSessionTimer)
        self.session_timer.start(1000)  # Update every second

        leftLayout = QVBoxLayout()

        # Create the Target Metrics box first
        self._createTargetMetricsBox(leftLayout)

        # Container for circle indicators and labels
        circleContainer = QHBoxLayout()
        
        # First Circle Indicator (Green)
        self.circleIndicator = QLabel(self)
        self.circleIndicator.setFixedSize(100, 100)  # Set the size of the circle
        self.circleIndicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._updateCircleIndicator(0)  # Initialize with default values

        # Label for Angle of Insertion (Green Circle)
        angleLabel = QLabel("Angle of Insertion")
        angleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        angleLabel.setStyleSheet("font-weight: bold;")  # Optional: Make the label bold

        # Vertical layout for Green Circle and its label
        greenCircleLayout = QVBoxLayout()
        greenCircleLayout.addWidget(self.circleIndicator)
        greenCircleLayout.addWidget(angleLabel)
        greenCircleLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        circleContainer.addLayout(greenCircleLayout)

        # Second Circle Indicator (Blue)
        self.circleIndicator2 = QLabel(self)
        self.circleIndicator2.setFixedSize(100, 100)  # Set the size of the circle
        self.circleIndicator2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._updateCircleIndicator2(0)  # Initialize with default values

        # Label for Elevation (Blue Circle)
        elevationLabel = QLabel("Elevation")
        elevationLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elevationLabel.setStyleSheet("font-weight: bold;")  # Optional: Make the label bold

        # Vertical layout for Blue Circle and its label
        blueCircleLayout = QVBoxLayout()
        blueCircleLayout.addWidget(self.circleIndicator2)
        blueCircleLayout.addWidget(elevationLabel)
        blueCircleLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        circleContainer.addLayout(blueCircleLayout)

        leftLayout.addLayout(circleContainer)

        # Rest of the UI setup remains unchanged...
        armImageLayout = QHBoxLayout()
        armImageLayout.addStretch()

        # Arm Image
        self.arm_image_label = QLabel(self)
        
        # Load the appropriate image based on the selected vein and insertion point
        if self.selected_vein == "Left Vein":
            if self.selected_point == "Point A":
                self.pixmap = QPixmap("Capstone/Feedback/0007.png")
            elif self.selected_point == "Point B":
                self.pixmap = QPixmap("Capstone/Feedback/middleleftvein-removebg-preview.png")
            elif self.selected_point == "Point C":
                self.pixmap = QPixmap("Capstone/Feedback/bottomleftvein-removebg-preview.png")
        elif self.selected_vein == "Right Vein":
            if self.selected_point == "Point A":
                self.pixmap = QPixmap("Capstone/Feedback/toprightvein-removebg-preview.png")
            elif self.selected_point == "Point B":
                self.pixmap = QPixmap("Capstone/Feedback/middlerightvein-removebg-preview.png")
            elif self.selected_point == "Point C":
                self.pixmap = QPixmap("Capstone/Feedback/bottomrightvein-removebg-preview.png")
        else:
            # Default image if no vein or point is selected (optional)
            self.pixmap = QPixmap("Capstone/Feedback/default_image.png")

        # Debug statement to confirm the image path
        print(f"Loading image: {self.pixmap}")

        # Ensure the pixmap is loaded successfully
        if self.pixmap.isNull():
            print(f"Error: Failed to load image for vein={self.selected_vein}, point={self.selected_point}")
            self.pixmap = QPixmap("Capstone/Feedback/default_image.png")  # Fallback to default image

        self.arm_image_label.setPixmap(self.pixmap.scaled(335, 475))
        self.arm_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        leftLayout.addWidget(self.arm_image_label)

        leftLayout.addLayout(armImageLayout)

        # Directional arrows (GIFs)
        self.arrow_up = QLabel(self.arm_image_label)
        self.arrow_up_movie = QMovie("Capstone/Feedback/arrow-1153-256-mainup-1-unscreen")
        self.arrow_up.setMovie(self.arrow_up_movie)
        self.arrow_up.setVisible(False)

        self.arrow_down = QLabel(self.arm_image_label)
        self.arrow_down_movie = QMovie("Capstone/Feedback/arrow-1153_256(maindown).gif")
        self.arrow_down.setMovie(self.arrow_down_movie)
        self.arrow_down.setVisible(False)

        self.arrow_left = QLabel(self.arm_image_label)
        self.arrow_left_movie = QMovie("Capstone/Feedback/arrow-358_256(mainleft).gif")
        self.arrow_left.setMovie(self.arrow_left_movie)
        self.arrow_left.setVisible(False)

        self.arrow_right = QLabel(self.arm_image_label)
        self.arrow_right_movie = QMovie("Capstone/Feedback/arrow-358-256-mainright--unscreen")
        self.arrow_right.setMovie(self.arrow_right_movie)
        self.arrow_right.setVisible(False)

        # Position Arrows (overlayed)
        self.arrow_up.move(400, 40)  # Adjust position as needed
        self.arrow_down.move(400, 235)
        self.arrow_left.move(220, 180)
        self.arrow_right.move(565, 180)

        self.arrow_up_movie.start()
        self.arrow_down_movie.start()
        self.arrow_left_movie.start()
        self.arrow_right_movie.start()

        # Needle Position, Angle, and Depth
        positionLayout = QGridLayout()
        positionLayout.addWidget(QLabel("Needle Position (x, y, z):"), 0, 0)
        self.positionInput = QLineEdit("(0.00, 0.00, 0.00)")
        positionLayout.addWidget(self.positionInput, 0, 1)

        positionLayout.addWidget(QLabel("Needle Angle (°):"), 1, 0)
        self.angleInput = QLineEdit("0.00")
        positionLayout.addWidget(self.angleInput, 1, 1)

        positionLayout.addWidget(QLabel("Needle Depth (mm):"), 2, 0)
        self.depthInput = QLineEdit("0.00")
        positionLayout.addWidget(self.depthInput, 2, 1)

        leftLayout.addLayout(positionLayout)

        # Guided Prompts and Warnings
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(QLabel("Guided Prompts and Warnings:"))
        self.promptsLog = QTextEdit()
        self.promptsLog.setReadOnly(True)
        rightLayout.addWidget(self.promptsLog)

        # Buttons
        buttonLayout = QHBoxLayout()

        self.endSimulationButton = QPushButton("End Simulation")

        # Style the button to be big and red
        self.endSimulationButton.setStyleSheet("""
            QPushButton {
                background-color: red;
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: white;
            }
        """)

        self.endSimulationButton.clicked.connect(self._endSimulation)
        buttonLayout.addWidget(self.endSimulationButton, alignment=Qt.AlignmentFlag.AlignCenter)

        rightLayout.addLayout(buttonLayout)

        # Add the vein plot to the top-right corner
        self._plotVeins(rightLayout)

        # Add layouts to the general layout
        self.generalLayout.addLayout(leftLayout, 1)  # Assign more weight to the left layout
        self.generalLayout.addLayout(rightLayout)

        self.session_time_label.raise_()

    def _updateSessionTimer(self):
        """Update the session timer each second."""
        self.elapsed_seconds += 1
        minutes = self.elapsed_seconds // 60
        seconds = self.elapsed_seconds % 60
        self.session_time_label.setText(f"{minutes:02d}:{seconds:02d}")

    def _updateCircleIndicator2(self, angle):
        """Update the second circle indicator with the current angle and color."""
        target_angle = float(self.targetAngle.text())
        deviation = abs(angle - target_angle)
        if deviation <= self.expert_pitch_std:
            color = QColor("green")
        elif deviation <= 1.5 * self.expert_pitch_std:
            color = QColor("orange")
        else:
            color = QColor("red")

        # Create a pixmap to draw the circle
        pixmap = QPixmap(self.circleIndicator2.size())
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, 100, 100)

        # Draw the angle text
        painter.setPen(QColor("white"))
        font = QFont()
        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"{angle:.1f}°")
        painter.end()

        self.circleIndicator2.setPixmap(pixmap)

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

        # Create a label to indicate the current view
        self.view_label = QLabel("", self.canvas)
        self.view_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                background-color: rgba(0, 0, 0, 150);
                border-radius: 5px;
                padding: 5px 10px;
            }
        """)
        self.view_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_label.setVisible(False)
        self.view_label.move(10, 10)  # Top-left corner of canvas

        # Plot the veins using the copied functions
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.grid(False)  # Remove the grid

        # Ensure the axis background is fully transparent
        self.ax.patch.set_alpha(0)

        # Hide axes and labels completely
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        for spine in self.ax.spines.values():
            spine.set_visible(False)  # Hide spines completely

        # Make all panes fully transparent
        self.ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

        # Make axis lines fully transparent
        self.ax.xaxis.line.set_alpha(0)
        self.ax.yaxis.line.set_alpha(0)
        self.ax.zaxis.line.set_alpha(0)

        # Get the mean trajectory, upper bound, and lower bound
        (mean_traj, upper_bound, lower_bound), trajectories = get_mean_std_bounds()

        # Plot the mean trajectory and bounds
        self.ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2], 'white', label="Mean")
        self.ax.plot(upper_bound[:, 0], upper_bound[:, 1], upper_bound[:, 2], 'yellow', label=f"Upper Bound (+{STD_ACCEPTABLE}σ)")
        self.ax.plot(lower_bound[:, 0], lower_bound[:, 1], lower_bound[:, 2], 'orange', label=f"Lower Bound (-{STD_ACCEPTABLE}σ)")

        # Initialize live trajectory (empty at first)
        self.live_trajectory = np.empty((0, 3))  # Empty array to store live data points
        self.live_line, = self.ax.plot([], [], [], 'deepskyblue', label="Live")  # Yellow line for live data

        # Remove title
        self.ax.set_title("")

        # Add a legend and move it outside the plot
        legend = self.ax.legend(loc='upper left', bbox_to_anchor=(0.8, 1), title="Legend")
        legend.set_visible(True)

        # Ensure the entire figure and axes background are transparent
        self.ax.set_facecolor('none')
        self.figure.patch.set_alpha(0)  # Make entire figure transparent

        # Store the axis limits from the mean trajectory
        self.xlim = self.ax.get_xlim()  # Store x-axis limits
        self.ylim = self.ax.get_ylim()  # Store y-axis limits
        self.zlim = self.ax.get_zlim()  # Store z-axis limits

        # Debug: Print the axis limits
        print(f"X-axis limits: {self.xlim}")
        print(f"Y-axis limits: {self.ylim}")
        print(f"Z-axis limits: {self.zlim}")

        # Set the initial view
        self.ax.view_init(elev=self.original_elev, azim=self.original_azim)

        self.view_toggle_state = True  # Start with XY view
        self.view_toggle_timer = QTimer()
        self.view_toggle_timer.timeout.connect(self._toggleView)
        self.view_toggle_timer.start(7000)  # Toggle every 5 seconds

        self.canvas.draw_idle()

        # Start a timer for live updates
        self.live_update_timer = QTimer()
        self.live_update_timer.timeout.connect(self._updateLiveTrajectory)
        self.live_update_timer.start(15)  # Update every 10 ms

    def _toggleView(self):
        """Start animating the transition between XY and YZ views and show the view label."""
        if self.view_toggle_state:
            self.start_elev, self.start_azim = 90, -90     # XY view (looking down Z)
            self.end_elev, self.end_azim = 0, 0            # YZ view (looking down X)
            view_text = "Side View"
        else:
            self.start_elev, self.start_azim = 0, 0
            self.end_elev, self.end_azim = 90, -90
            view_text = "Top View"

        self.animation_step = 0
        self.animation_steps = 20
        self.view_toggle_state = not self.view_toggle_state

        # Show and fade in the view label
        self._showViewLabel(view_text)

        # Animate the view transition
        self.view_animation_timer = QTimer()
        self.view_animation_timer.timeout.connect(self._animateViewTransition)
        self.view_animation_timer.start(25)

    def _showViewLabel(self, text):
        """Display a view label that fades in and out."""
        self.view_label.setText(text)
        self.view_label.setVisible(True)
        self.view_label.raise_()

        self.view_label_opacity = QGraphicsOpacityEffect()
        self.view_label.setGraphicsEffect(self.view_label_opacity)
        self.view_label_opacity.setOpacity(0.0)

        self.label_fade_in_timer = QTimer()
        self.label_fade_out_timer = QTimer()
        self.label_fade_step = 0

        def fade_in():
            opacity = self.label_fade_step / 10
            self.view_label_opacity.setOpacity(opacity)
            self.label_fade_step += 1
            if self.label_fade_step > 10:
                self.label_fade_in_timer.stop()
                QTimer.singleShot(1500, fade_out)  # Stay visible before fading out

        def fade_out():
            self.label_fade_step = 10
            self.label_fade_out_timer.timeout.connect(step_out)
            self.label_fade_out_timer.start(50)

        def step_out():
            opacity = self.label_fade_step / 10
            self.view_label_opacity.setOpacity(opacity)
            self.label_fade_step -= 1
            if self.label_fade_step < 0:
                self.label_fade_out_timer.stop()
                self.view_label.setVisible(False)

        self.label_fade_in_timer.timeout.connect(fade_in)
        self.label_fade_in_timer.start(50)

    def _animateViewTransition(self):
        """Animate each step of the view transition."""
        t = self.animation_step / self.animation_steps  # Normalized time [0,1]
        elev = (1 - t) * self.start_elev + t * self.end_elev
        azim = (1 - t) * self.start_azim + t * self.end_azim

        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

        self.animation_step += 1
        if self.animation_step > self.animation_steps:
            self.view_animation_timer.stop()


    def _updateLiveTrajectory(self):
        """Update the live trajectory using the last 4 points (current + previous 3)."""
        if self.live_trajectory_queue.empty():
            return

        new_point = self.live_trajectory_queue.get()

        # Normalize the new point to fit within the axis limits
        new_point[0] = np.clip(new_point[0], self.xlim[0], self.xlim[1])
        new_point[1] = np.clip(new_point[1], self.ylim[0], self.ylim[1])
        new_point[2] = np.clip(new_point[2], self.zlim[0], self.zlim[1])

        # Initialize or update the rolling buffer of points
        if not hasattr(self, 'live_trajectory_buffer'):
            self.live_trajectory_buffer = []

        self.live_trajectory_buffer.append(new_point)

        # Keep only the last 4 points
        if len(self.live_trajectory_buffer) > 4:
            self.live_trajectory_buffer.pop(0)

        # Convert to numpy array for plotting
        trajectory_array = np.array(self.live_trajectory_buffer)

        # Clear previous line and point if they exist
        if hasattr(self, 'live_line'):
            self.live_line.remove()
        if hasattr(self, 'live_point'):
            self.live_point.remove()

        # Plot the line through the last 4 points
        self.live_line, = self.ax.plot(
            trajectory_array[:, 0],
            trajectory_array[:, 1],
            trajectory_array[:, 2],
            'deepskyblue',  # deepskyblue line
            linewidth=2,
            label="Live"
        )

        # Plot the most recent point
        self.live_point = self.ax.scatter(
            new_point[0],
            new_point[1],
            new_point[2],
            color='deepskyblue',
            s=50,
            label="Current"
        )

        # Redraw the canvas
        self.canvas.draw_idle()

    def _animatePlot(self):
        """Animate the plot by rotating and panning, then return to the original position."""
        if not hasattr(self, 'animation_step'):
            # Initialize animation parameters
            self.animation_step = 0
            self.animation_duration = 100  # Number of steps for the animation
            self.animation_timer.setInterval(100)  # Update every 50 ms

        if self.animation_step < self.animation_duration:
            # Calculate intermediate values for elevation, azimuth, and panning
            progress = self.animation_step / self.animation_duration
            elev = self.original_elev + 30 * np.sin(2 * np.pi * progress)  # Oscillate elevation
            azim = self.original_azim + 360 * progress  # Rotate azimuth
            x_shift = 10 * np.sin(2 * np.pi * progress)  # Oscillate x-axis panning
            y_shift = 10 * np.cos(2 * np.pi * progress)  # Oscillate y-axis panning

            # Update the view
            self.ax.view_init(elev=elev, azim=azim)
            self.ax.set_xlim([self.original_xlim[0] + x_shift, self.original_xlim[1] + x_shift])
            self.ax.set_ylim([self.original_ylim[0] + y_shift, self.original_ylim[1] + y_shift])

            # Redraw the canvas
            self.canvas.draw_idle()

            self.animation_step += 1
        else:
            # Reset to the original position
            self.ax.view_init(elev=self.original_elev, azim=self.original_azim)
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.ax.set_zlim(self.original_zlim)
            self.canvas.draw_idle()

            # Stop the timer
            self.animation_timer.stop()
            del self.animation_step  # Reset animation state

    def _createTargetMetricsBox(self, layout):
        """Create the Target Metrics box."""
        self.groupBox = QGroupBox("Target Metrics")
        layout_metrics = QGridLayout()

        layout_metrics.addWidget(QLabel("Target Depth (mm):"), 0, 0)
        self.targetDepth = QLineEdit(str(self.expert_yaw))
        layout_metrics.addWidget(self.targetDepth, 0, 1)

        layout_metrics.addWidget(QLabel("Target Angle (°):"), 1, 0)
        self.targetAngle = QLineEdit(str(self.expert_pitch))
        layout_metrics.addWidget(self.targetAngle, 1, 1)

        self.groupBox.setLayout(layout_metrics)
        layout.addWidget(self.groupBox)

        # Hide the group box
        self.groupBox.setVisible(False)

    def _updateCircleIndicator(self, angle):
        """Update the circle indicator with the current angle and color."""
        target_angle = float(self.targetDepth.text())
        deviation = abs(angle - target_angle)
        if deviation <= self.expert_yaw_std:
            color = QColor("green")
        elif deviation <= 1.5 * self.expert_yaw_std:
            color = QColor("orange")
        else:
            color = QColor("red")

        # Create a pixmap to draw the circle
        pixmap = QPixmap(self.circleIndicator.size())
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, 100, 100)

        # Draw the angle text
        painter.setPen(QColor("white"))
        font = QFont()
        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"{angle:.1f}°")
        painter.end()

        self.circleIndicator.setPixmap(pixmap)

    def _updateData(self):
        """Simulate data updates and display feedback for needle guidance."""
        if not self.work_queue.empty():
            data = self.work_queue.get()
            if data is None:
                self.timer.stop()
                self.promptsLog.append("Simulation complete.")
                self._showSummaryPage()
                return
        else:
            return
        
        if not self.direction_intruction_queue.empty():
            direction = self.direction_intruction_queue.get()
        else:
            direction = None
        
        x, y, z, pitch, roll, yaw = data
        self.live_trajectory_queue.put([x, y, z])

        simulated_position = f"({x:.2f}, {y:.2f}, {z:.2f})"
        simulated_elevation = pitch
        simulated_angle_of_insertion = yaw

        self.positionInput.setText(simulated_position)
        self.angleInput.setText(f"{simulated_elevation:.2f}")
        self.depthInput.setText(f"{simulated_angle_of_insertion:.2f}")

        feedback = self._generateFeedback(direction, simulated_angle_of_insertion, simulated_elevation)
        self.promptsLog.append(f"Update {self.update_count + 1}: {feedback}")

        target_angle = float(self.targetAngle.text())
        target_depth = float(self.targetDepth.text())
        self.total_angle_deviation += abs(simulated_elevation - target_angle)
        self.total_depth_deviation += abs(simulated_angle_of_insertion - target_depth)

        # Update the circle indicator
        self._updateCircleIndicator(simulated_angle_of_insertion)
        self._updateCircleIndicator2(simulated_elevation)

        self.update_count += 1

    def _setArrowVisibility(self, direction, is_visible):
        """Control visibility of directional arrows based on conditions."""
        arrow_dict = {
            "up": self.arrow_up,
            "down": self.arrow_down,
            "left": self.arrow_left,
            "right": self.arrow_right
        }
        arrow = arrow_dict.get(direction)
        if arrow:
            arrow.setVisible(is_visible)

    def _generateFeedback(self, direction, angle, depth):
        """Generate feedback based on current and target metrics."""
        target_angle = float(self.targetAngle.text())
        target_depth = float(self.targetDepth.text())

        feedback = []

        self.arrow_up.setVisible(False)
        self.arrow_down.setVisible(False)
        self.arrow_left.setVisible(False)
        self.arrow_right.setVisible(False)
        
        if direction is not None:
            for item in direction:
                if item == "left":
                    feedback.append("Move the IV to the left.")
                    self.arrow_left.setVisible(True)
                    print("Move the IV to the left.")
                if item == "right":
                    feedback.append("Move the IV to the right.")
                    self.arrow_right.setVisible(True)
                if item == "up":
                    feedback.append("Move the IV forward.")
                    self.arrow_up.setVisible(True)
                if item == "down":
                    feedback.append("Move the IV backward.")
                    self.arrow_down.setVisible(True)

        return "\n".join(feedback) + "\n" + "-" * 40

    def _restartSimulation(self):
        """Restart the simulation."""
        self.positionInput.setText("(0.00, 0.00, 0.00)")
        self.angleInput.setText("0.00")
        self.depthInput.setText("0.00")
        self.promptsLog.clear()
        self.update_count = 0
        self.total_angle_deviation = 0
        self.total_depth_deviation = 0
        self.timer.start(self.timer.interval())

        self.arrow_up.setVisible(False)
        self.arrow_down.setVisible(False)
        self.arrow_left.setVisible(False)
        self.arrow_right.setVisible(False)

    def _logSessionData(self):
        """Log the current session data to a file."""
        position = self.positionInput.text()
        angle = self.angleInput.text()
        depth = self.depthInput.text()
        session_data = f"Position: {position}, Angle: {angle}, Depth: {depth}\n"

        self.sessionLog.append(session_data)
        with open("session_log.txt", "a") as file:
            file.write(session_data)

    def _endSimulation(self):
        """End the simulation and show the summary page."""
        self.timer.stop()
        self._showSummaryPage()

    def _showSummaryPage(self):
        """Display the summary of the simulation."""
        angle_error = self.total_angle_deviation / self.max_updates
        depth_error = self.total_depth_deviation / self.max_updates

        score = max(0, 10 - (angle_error + depth_error) / 10)
        score = round(score, 2)

        if score == 10:
            feedback = ("Excellent performance! Your actions demonstrate a high level of accuracy and precision." 
                       "Maintain this level of focus and attention to detail in future tasks. Great job!")
        elif score >= 7:
            feedback = ("Good performance! You show a solid understanding of the task, but there are occasional minor errors."
                        "To improve further, double-check your movements or decisions to ensure consistency. " 
                        "Consider reviewing any specific steps where you felt less confident.")
        elif score >= 5:
            feedback = (
                "Fair performance. While you understand the basics, there are noticeable areas of inconsistency. "
                "Analyze where mistakes occurred—was it in positioning, timing, or precision? "
                "Practice those specific aspects, and seek feedback on how to refine your technique."
            )
        else:
            feedback = (
                "Needs improvement. It seems there were significant challenges in accuracy or approach. "
                "Take time to revisit the fundamental concepts and techniques. "
                "Break the task into smaller steps, practice each one thoroughly, and don't hesitate to ask for guidance."
            )

        dialog = QDialog(self)
        dialog.setWindowTitle("Simulation Summary")
        dialog_layout = QVBoxLayout()

        dialog_layout.addWidget(QLabel(f"Final Score: {score}/10"))
        dialog_layout.addWidget(QLabel(feedback))

        # Button to return to the IntroScreen
        returnToIntroButton = QPushButton("Return to Start")
        returnToIntroButton.clicked.connect(lambda: self._returnToIntroScreen(dialog))
        dialog_layout.addWidget(returnToIntroButton)

        self.app_to_signal_processing.put([None, None])
        print(f"{list(self.app_to_signal_processing.queue)=}")

        dialog.setLayout(dialog_layout)
        dialog.exec()

    def _returnToIntroScreen(self, dialog):
        """Return to the IntroScreen and restart the application flow."""
        dialog.close()  # Close the summary dialog
        self.close()    # Close the FeedbackUI window

        # Re-launch the intro screen
        intro_screen = IntroScreen()
        if intro_screen.exec() == QDialog.DialogCode.Accepted:
            vein_screen = PickVeinScreen()
            if vein_screen.exec() == QDialog.DialogCode.Accepted:
                selected_vein = vein_screen.selected_vein

                insertion_point_screen = PickInsertionPointScreen(selected_vein)
                if insertion_point_screen.exec() == QDialog.DialogCode.Accepted:
                    selected_insertion_point = insertion_point_screen.selected_point

                    self.app_to_signal_processing.put([selected_vein, selected_insertion_point])
                    print(f"{list(self.app_to_signal_processing.queue)=}")

                    # Relaunch Feedback UI
                    self.feedback_ui = FeedbackUI(selected_vein,
                                                  selected_insertion_point,
                                                  work_queue=self.work_queue,
                                                  angle_range_queue=self.angle_range_queue,
                                                  app_to_signal_processing=self.app_to_signal_processing,
                                                  direction_intruction_queue=self.direction_intruction_queue)
                    self.feedback_ui.show()

class MainApplication:
    """Manages the flow of the application."""
    def __init__(self, sig_processed_queue, app_to_signal_processing, angle_range_queue, direction_intruction_queue):
        self.sig_processed_queue = sig_processed_queue
        self.app_to_signal_processing = app_to_signal_processing
        self.angle_range_queue = angle_range_queue
        self.direction_intruction_queue = direction_intruction_queue
        self.app = QApplication(sys.argv)
        self.selected_vein = None
        self.selected_insertion_point = None

    def run(self):
        # Intro Screen
        intro_screen = IntroScreen()
        if intro_screen.exec() == QDialog.DialogCode.Accepted:
            # Pick Vein Screen
            vein_screen = PickVeinScreen()
            if vein_screen.exec() == QDialog.DialogCode.Accepted:
                self.selected_vein = vein_screen.selected_vein
                print(f"Selected Vein: {self.selected_vein}")  # Debug statement
                

                # Pick Insertion Point Screen
                insertion_point_screen = PickInsertionPointScreen(self.selected_vein)
                if insertion_point_screen.exec() == QDialog.DialogCode.Accepted:
                    self.selected_insertion_point = insertion_point_screen.selected_point
                    print(f"Selected Insertion Point: {self.selected_insertion_point}")  # Debug statement

                    self.app_to_signal_processing.put([self.selected_vein, self.selected_insertion_point])
                    print(f"{list(self.app_to_signal_processing.queue)=}")

                    # Launch Feedback UI with selected vein and insertion point
                    feedback_ui = FeedbackUI(self.selected_vein,
                                             self.selected_insertion_point,
                                             work_queue=self.sig_processed_queue,
                                             angle_range_queue=self.angle_range_queue,
                                             app_to_signal_processing=self.app_to_signal_processing,
                                             direction_intruction_queue = self.direction_intruction_queue)
                    feedback_ui.show()
                    sys.exit(self.app.exec())

if __name__ == "__main__":
    sig_processed = Queue()
    app_to_signal_processing = Queue()
    angle_range_queue = Queue()
    main_app = MainApplication(sig_processed, app_to_signal_processing, angle_range_queue)
    main_app.run()