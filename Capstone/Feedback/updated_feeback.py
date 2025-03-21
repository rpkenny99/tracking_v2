from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QMovie, QFont, QColor, QPainter
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel,
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
    def __init__(self, selected_vein, selected_point, max_updates=12, update_interval=10, work_queue=None):
        super().__init__()
        self.setWindowTitle("Feedback UI - Needle Insertion")
        self.showFullScreen()  # Make the window maximized

        # Store the selected vein and insertion point
        self.selected_vein = selected_vein
        self.selected_point = selected_point
        self.work_queue = work_queue

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
        self.timer.start(update_interval)

    def _createDisplay(self):
        """Create the UI layout with the updated design."""
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
        self.restartButton = QPushButton("Restart")
        self.restartButton.clicked.connect(self._restartSimulation)
        buttonLayout.addWidget(self.restartButton)

        self.logButton = QPushButton("Log Data")
        self.logButton.clicked.connect(self._logSessionData)
        buttonLayout.addWidget(self.logButton)

        rightLayout.addLayout(buttonLayout)

        # Add the vein plot to the top-right corner
        self._plotVeins(rightLayout)

        # Add layouts to the general layout
        self.generalLayout.addLayout(leftLayout, 1)  # Assign more weight to the left layout
        self.generalLayout.addLayout(rightLayout)

    def _updateCircleIndicator2(self, angle):
        """Update the second circle indicator with the current angle and color."""
        target_angle = float(self.targetAngle.text())
        deviation = abs(angle - target_angle)
        color = QColor("blue") if deviation <= 1.5 else QColor("orange")  # Use different colors for distinction

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

        # Plot the veins
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

        # Load and plot left vein
        left_vein_file = "Capstone/Feedback/leftveinvein2_smoothed4.xlsx"
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
        self.ax.plot(points_left[:, 0], points_left[:, 1], points_left[:, 2], 'b', label="Left Vein")

        # Load and plot right vein
        right_vein_file = "Capstone/Feedback/rightvein2.xlsx"
        right_data = pd.read_excel(right_vein_file)
        Tx_right, Ty_right, Tz_right = right_data['Tx'].to_numpy(), right_data['Ty'].to_numpy(), right_data['Tz'].to_numpy()
        valid_indices_right = (Tx_right > threshold) & (Ty_right > threshold) & (Tz_right > threshold)
        Tx_right, Ty_right, Tz_right = Tx_right[valid_indices_right], Ty_right[valid_indices_right], Tz_right[valid_indices_right]

        # Scale the Tx, Ty, and Tz values by 2 to stretch the veins along all axes
        Tx_right = Tx_right * 2
        Ty_right = Ty_right * 2
        Tz_right = Tz_right * 2

        points_right = np.vstack((Tx_right, Ty_right, Tz_right)).T
        self.ax.plot(points_right[:, 0], points_right[:, 1], points_right[:, 2], 'r', label="Right Vein")

        # Remove title and legend
        self.ax.set_title("")
        self.ax.legend().set_visible(False)

        # Ensure the entire figure and axes background are transparent
        self.ax.set_facecolor('none')
        self.figure.patch.set_alpha(0)  # Make entire figure transparent

        # Store the original view and axis limits
        self.original_elev = 30  # Default elevation
        self.original_azim = -60  # Default azimuth
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        self.original_zlim = self.ax.get_zlim()

        # Set the initial view
        self.ax.view_init(elev=self.original_elev, azim=self.original_azim)
        self.canvas.draw()

        # Start the animation after 2 seconds
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._animatePlot)
        self.animation_timer.start(2000)  # 2-second delay

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
            self.canvas.draw()

            self.animation_step += 1
        else:
            # Reset to the original position
            self.ax.view_init(elev=self.original_elev, azim=self.original_azim)
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.ax.set_zlim(self.original_zlim)
            self.canvas.draw()

            # Stop the timer
            self.animation_timer.stop()
            del self.animation_step  # Reset animation state

    def _createTargetMetricsBox(self, layout):
        """Create the Target Metrics box."""
        self.groupBox = QGroupBox("Target Metrics")
        layout_metrics = QGridLayout()

        layout_metrics.addWidget(QLabel("Target Depth (mm):"), 0, 0)
        self.targetDepth = QLineEdit("50.00")
        layout_metrics.addWidget(self.targetDepth, 0, 1)

        layout_metrics.addWidget(QLabel("Target Angle (°):"), 1, 0)
        self.targetAngle = QLineEdit("30.00")
        layout_metrics.addWidget(self.targetAngle, 1, 1)

        self.groupBox.setLayout(layout_metrics)
        layout.addWidget(self.groupBox)

        # Hide the group box
        self.groupBox.setVisible(False)

    def _updateCircleIndicator(self, angle):
        """Update the circle indicator with the current angle and color."""
        target_angle = float(self.targetAngle.text())
        deviation = abs(angle - target_angle)
        color = QColor("green") if deviation <= 1.5 else QColor("red")

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
        
        x, y, z, pitch, roll, yaw = data

        simulated_position = f"({x:.2f}, {y:.2f}, {z:.2f})"
        simulated_elevation = pitch
        simulated_angle_of_insertion = yaw

        self.positionInput.setText(simulated_position)
        self.angleInput.setText(f"{simulated_elevation:.2f}")
        self.depthInput.setText(f"{simulated_angle_of_insertion:.2f}")

        feedback = self._generateFeedback(simulated_position, simulated_angle_of_insertion, simulated_elevation)
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

    def _generateFeedback(self, position, angle, depth):
        """Generate feedback based on current and target metrics."""
        target_angle = float(self.targetAngle.text())
        target_depth = float(self.targetDepth.text())
        x = float(position.strip("()").split(",")[0])

        feedback = []

        self.arrow_up.setVisible(False)
        self.arrow_down.setVisible(False)
        self.arrow_left.setVisible(False)
        self.arrow_right.setVisible(False)

        if angle < target_angle and x < 0:
            feedback.append("Tilt the IV upwards.")
            self.arrow_up.setVisible(True)
        elif angle > target_angle:
            feedback.append("Tilt the IV downwards.")
            self.arrow_down.setVisible(True)
        
        if x < 0:
            feedback.append("Tilt the IV to the left.")
            self.arrow_left.setVisible(True)
        elif x > 0:
            feedback.append("Tilt the IV to the right.")
            self.arrow_right.setVisible(True)

        if depth < target_depth:
            feedback.append("Insert the needle a bit deeper.")
        elif depth > target_depth:
            feedback.append("Pull the needle out slightly.")

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

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        dialog_layout.addWidget(close_button)

        dialog.setLayout(dialog_layout)
        dialog.exec()

class MainApplication:
    """Manages the flow of the application."""
    def __init__(self, sig_processed_queue, app_to_signal_processing):
        self.sig_processed_queue = sig_processed_queue
        self.app_to_signal_processing = app_to_signal_processing
        self.app = QApplication(sys.argv)
        self.selected_vein = None
        self.selected_insertion_point = None

    def run(self):
        # Intro Screen
        intro_screen = IntroScreen()
        if intro_screen.exec() == QDialog.DialogCode.Accepted:
            self.app_to_signal_processing.put([1, None, None])
            # Pick Vein Screen
            vein_screen = PickVeinScreen()
            if vein_screen.exec() == QDialog.DialogCode.Accepted:
                self.selected_vein = vein_screen.selected_vein
                print(f"Selected Vein: {self.selected_vein}")  # Debug statement

                self.app_to_signal_processing.put([1, self.selected_vein, None])
                

                # Pick Insertion Point Screen
                insertion_point_screen = PickInsertionPointScreen(self.selected_vein)
                if insertion_point_screen.exec() == QDialog.DialogCode.Accepted:
                    self.selected_insertion_point = insertion_point_screen.selected_point
                    print(f"Selected Insertion Point: {self.selected_insertion_point}")  # Debug statement

                    self.app_to_signal_processing.put([1, self.selected_vein, self.selected_insertion_point])

                    # Launch Feedback UI with selected vein and insertion point
                    feedback_ui = FeedbackUI(self.selected_vein, self.selected_insertion_point, work_queue=self.sig_processed_queue)
                    feedback_ui.show()
                    sys.exit(self.app.exec())

if __name__ == "__main__":
    sig_processed = Queue()
    main_app = MainApplication(sig_processed)
    main_app.run()