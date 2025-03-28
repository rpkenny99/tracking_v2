from PyQt6.QtCore import QTimer, Qt, QSize
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
from PyQt6.QtGui import QPixmap, QTransform
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
# Hide debug messages (only show warnings and above)
matplotlib.set_loglevel("warning")

JET_BLACK_STYLE = """
    QWidget {
        background-color: #121212;  /* Jet black */
        color: white;  /* Default text color */
    }
    
    QLabel {
        color: white;
    }
    
    QTextEdit, QLineEdit {
        background-color: #1E1E1E;
        color: white;
        border: 1px solid #333;
    }
    
    QGroupBox {
        border: 1px solid #333;
        border-radius: 5px;
        margin-top: 10px;
    }
    
    QGroupBox::title {
        color: white;
        subcontrol-origin: margin;
        left: 10px;
    }
"""

class IntroScreen(QDialog):
    """Introductory screen to start the simulation."""
    def __init__(self):
        super().__init__()
        self.setStyleSheet(JET_BLACK_STYLE)
        self.setWindowTitle("Cyber-Physical Infant IV Simulator")
        self.setFixedSize(1366, 700)  # Make the window maximized

        layout = QVBoxLayout()
        label = QLabel("Cyber-Physical Infant IV Simulator")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        font = QFont()
        font.setPointSize(26)
        font.setBold(True)
        label.setFont(font)

        layout.addWidget(label)

        start_button = QPushButton("Start Simulation")
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;  /* Sky blue */
                color: black;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin: 10px;
                border: 2px solid #4682B4;  /* Steel blue border */
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #B0E2FF;  /* Light sky blue */
                border: 2px solid #87CEEB;
            }
            QPushButton:pressed {
                background-color: #4682B4;  /* Steel blue */
                color: white;
            }
        """)
        start_button.clicked.connect(self.accept)  # Close the dialog on click
        layout.addWidget(start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Exit Button (Red)
        exit_button = QPushButton("Exit")
        exit_button.setStyleSheet("""
            QPushButton {
                background-color: #FF3333;  /* Bright red */
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin: 10px;
                border: 2px solid #CC0000;  /* Darker red border */
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #FF6666;  /* Lighter red */
                border: 2px solid #FF3333;
            }
            QPushButton:pressed {
                background-color: #CC0000;  /* Darker red */
            }
        """)
        exit_button.clicked.connect(self.reject)  # Close the app
        layout.addWidget(exit_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)
        self.setStyleSheet("background-color: black; color: white;")
        self.setStyleSheet("""
            QPushButton {
                border: 2px solid white;
                color: white;
                background-color: black;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
            }
            QWidget {
                background-color: black;
                color: white;
            }
        """)

class PickVeinScreen(QDialog):
    """Choose the Vein to be pierced."""
    def __init__(self):
        super().__init__()
        self.setStyleSheet(JET_BLACK_STYLE)
        self.setWindowTitle("Pick Your Vein")
        self.setFixedSize(1366, 700)
        
        self.setStyleSheet("background-color: black; color: white;")
        self.setStyleSheet("""
            QPushButton {
                border: 2px solid white;
                color: white;
                background-color: black;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
            }
            QWidget {
                background-color: black;
                color: white;
            }
        """)

        # Main layout
        main_layout = QVBoxLayout()

        # Top bar layout (Go Back + Title)
        top_bar_layout = QHBoxLayout()
        
        # Go Back Button (Left-aligned)
        back_button = QPushButton("← Go Back")
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #FFD700;  /* Bright yellow */
                color: black;              /* Black text for better contrast */
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin-top: 10px;
                border: 2px solid #DAA520; /* Gold border */
            }
            QPushButton:hover {
                background-color: #FFEC8B;  /* Lighter yellow on hover */
                border: 2px solid #FFD700;
            }
            QPushButton:pressed {
                background-color: #DAA520;  /* Darker yellow when pressed */
            }
        """)
        back_button.clicked.connect(self.go_back)
        top_bar_layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignLeft)

        # Left stretch (pushes title right)
        top_bar_layout.addStretch()

        # Title (Centered)
        header_label = QLabel("Pick Your Vein")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        header_label.setFont(font)
        top_bar_layout.addWidget(header_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Right stretch (pushes title left)
        top_bar_layout.addStretch()

        main_layout.addLayout(top_bar_layout)

        # Rest of the UI (image, buttons, etc.)
        self.arm_image_label = QLabel(self)
        self.pixmap = QPixmap("Capstone/Feedback/redv3in.png")
        transform = QTransform().rotate(180)
        self.pixmap = self.pixmap.transformed(transform)
        self.arm_image_label.setPixmap(self.pixmap.scaled(950, 450, Qt.AspectRatioMode.KeepAspectRatio))
        self.arm_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.arm_image_label)

        # Vein Selection Buttons
        button_layout = QHBoxLayout()
        left_vein_button = QPushButton("Left Vein")
        left_vein_button.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;  /* Sky blue */
                color: black;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin: 5px;
                border: 2px solid #4682B4;  /* Steel blue border */
            }
            QPushButton:hover {
                background-color: #B0E2FF;  /* Light sky blue */
                border: 2px solid #87CEEB;
            }
            QPushButton:pressed {
                background-color: #4682B4;  /* Steel blue */
                color: white;
            }
        """)
        left_vein_button.clicked.connect(self.select_left_vein)
        button_layout.addWidget(left_vein_button)

        right_vein_button = QPushButton("Right Vein")
        right_vein_button.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;  /* Sky blue */
                color: black;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin: 5px;
                border: 2px solid #4682B4;  /* Steel blue border */
            }
            QPushButton:hover {
                background-color: #B0E2FF;  /* Light sky blue */
                border: 2px solid #87CEEB;
            }
            QPushButton:pressed {
                background-color: #4682B4;  /* Steel blue */
                color: white;
            }
        """)
        right_vein_button.clicked.connect(self.select_right_vein)
        button_layout.addWidget(right_vein_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def select_left_vein(self):
        self.selected_vein = "Left Vein"
        self.accept()

    def select_right_vein(self):
        self.selected_vein = "Right Vein"
        self.accept()

    def go_back(self):
        self.reject()

class PickInsertionPointScreen(QDialog):
    def __init__(self, selected_vein):
        super().__init__()
        self.setStyleSheet(JET_BLACK_STYLE)
        self.setWindowTitle("Pick Insertion Point")
        self.setFixedSize(1366, 700)
        self.selected_vein = selected_vein

        # Main layout
        main_layout = QVBoxLayout()

        # Top bar layout (Go Back + Title)
        top_bar_layout = QHBoxLayout()
        
        # Go Back Button (Left-aligned)
        back_button = QPushButton("← Go Back")
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #FFD700;  /* Bright yellow */
                color: black;              /* Black text for better contrast */
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin-top: 10px;
                border: 2px solid #DAA520; /* Gold border */
            }
            QPushButton:hover {
                background-color: #FFEC8B;  /* Lighter yellow on hover */
                border: 2px solid #FFD700;
            }
            QPushButton:pressed {
                background-color: #DAA520;  /* Darker yellow when pressed */
            }
        """)
        back_button.clicked.connect(self.go_back)
        top_bar_layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignLeft)

        # Add stretch to push title to center
        top_bar_layout.addStretch()

        # Title (Centered)
        header_label = QLabel("Pick Insertion Point")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        header_label.setFont(font)
        top_bar_layout.addWidget(header_label)

        # Add another stretch to balance the layout
        top_bar_layout.addStretch()

        main_layout.addLayout(top_bar_layout)

        # Arm Image
        self.setStyleSheet("background-color: black; color: white;")
        self.setStyleSheet("""
            QPushButton {
                border: 2px solid white;
                color: white;
                background-color: black;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
            }
            QWidget {
                background-color: black;
                color: white;
            }
        """)

        # Arm Image with Clickable Points
        self.arm_image_label = QLabel(self)
        if self.selected_vein == "Left Vein":
            self.pixmap = QPixmap("Capstone/Feedback/leftvein-removebg-preview.png")
        elif self.selected_vein == "Right Vein":
            self.pixmap = QPixmap("Capstone/Feedback/rightvein-removebg-preview.png")
        transform = QTransform().rotate(180)
        self.pixmap = self.pixmap.transformed(transform)
        self.arm_image_label.setPixmap(self.pixmap.scaled(950, 450, Qt.AspectRatioMode.KeepAspectRatio))
        self.arm_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.arm_image_label)

        # Insertion Point Buttons
        button_layout = QHBoxLayout()
        point_a_button = QPushButton("Top")
        point_a_button.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;  /* Sky blue */
                color: black;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin: 5px;
                border: 2px solid #4682B4;  /* Steel blue border */
            }
            QPushButton:hover {
                background-color: #B0E2FF;  /* Light sky blue */
                border: 2px solid #87CEEB;
            }
            QPushButton:pressed {
                background-color: #4682B4;  /* Steel blue */
                color: white;
            }
        """)
        point_a_button.clicked.connect(self.select_point_a)
        button_layout.addWidget(point_a_button)

        point_b_button = QPushButton("Middle")
        point_b_button.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;  /* Sky blue */
                color: black;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin: 5px;
                border: 2px solid #4682B4;  /* Steel blue border */
            }
            QPushButton:hover {
                background-color: #B0E2FF;  /* Light sky blue */
                border: 2px solid #87CEEB;
            }
            QPushButton:pressed {
                background-color: #4682B4;  /* Steel blue */
                color: white;
            }
        """)
        point_b_button.clicked.connect(self.select_point_b)
        button_layout.addWidget(point_b_button)

        point_c_button = QPushButton("Bottom")
        point_c_button.setStyleSheet("""
            QPushButton {
                background-color: #87CEEB;  /* Sky blue */
                color: black;
                font-size: 20px;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 10px;
                margin: 5px;
                border: 2px solid #4682B4;  /* Steel blue border */
            }
            QPushButton:hover {
                background-color: #B0E2FF;  /* Light sky blue */
                border: 2px solid #87CEEB;
            }
            QPushButton:pressed {
                background-color: #4682B4;  /* Steel blue */
                color: white;
            }
        """)
        point_c_button.clicked.connect(self.select_point_c)
        button_layout.addWidget(point_c_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def select_point_a(self):
        self.selected_point = "Point A"
        self.accept()

    def select_point_b(self):
        self.selected_point = "Point B"
        self.accept()

    def select_point_c(self):
        self.selected_point = "Point C"
        self.accept()

    def go_back(self):
        self.reject()  # Return to PickVeinScreen

class FeedbackUI(QMainWindow):
    def __init__(self,
                 selected_vein,
                 selected_point,
                 max_updates=12,
                 update_interval=5,
                 work_queue=None,
                 angle_range_queue=None,
                 app_to_signal_processing=None,
                 direction_intruction_queue = None,
                 user_score_queue=None):
        super().__init__()
        self.setStyleSheet(JET_BLACK_STYLE)
        self.setWindowTitle("Feedback UI - Needle Insertion")
        self.showFullScreen()  # Make the window maximized

        self.live_trajectory_queue = Queue()
        
        self.setStyleSheet("background-color: black; color: white;")
        self.setStyleSheet("""
            QPushButton {
                border: 2px solid white;
                color: white;
                background-color: black;
            }
            QPushButton:hover {
                background-color: white;
                color: black;
            }
            QWidget {
                background-color: black;
                color: white;
            }
        """)

        # Store the selected vein and insertion point
        self.selected_vein = selected_vein
        self.selected_point = selected_point
        self.work_queue = work_queue
        self.angle_range_queue = angle_range_queue
        self.app_to_signal_processing = app_to_signal_processing
        self.direction_intruction_queue = direction_intruction_queue
        self.user_score_queue = user_score_queue

        self.expert_pitch, _, self.expert_yaw, self.expert_pitch_std, _, self.expert_yaw_std = angle_range_queue.get()

        # Set this for top-down perspective
        self.original_elev = 90
        self.original_azim = -90

        


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

        # ---- Modified Circle Indicators Section ----
        leftLayout.addSpacing(70)
        # Vertical layout for both circles (aligned left)
        circleVerticalLayout = QVBoxLayout()
        circleVerticalLayout.setSpacing(10)  # Space between circles
        circleVerticalLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)  # <-- Key change

        # --- Angle of Insertion (Top Circle) ---
        self.circleIndicator = QLabel(self)
        self.circleIndicator.setFixedSize(100, 100)
        self.circleIndicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._updateCircleIndicator(0)

        angleLabel = QLabel("Angle of Insertion")
        angleLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)  # <-- Align text left
        angleLabel.setStyleSheet("font-weight: bold; padding-left: 5px;")  # <-- Add slight padding

        # --- Elevation (Bottom Circle) ---
        self.circleIndicator2 = QLabel(self)
        self.circleIndicator2.setFixedSize(100, 100)
        self.circleIndicator2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._updateCircleIndicator2(0)

        elevationLabel = QLabel("Elevation")
        elevationLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)  # <-- Align text left
        elevationLabel.setStyleSheet("font-weight: bold; padding-left: 5px;")  # <-- Add slight padding

        # Add both circles to vertical layout
        circleVerticalLayout.addWidget(self.circleIndicator)
        circleVerticalLayout.addWidget(angleLabel)
        circleVerticalLayout.addWidget(self.circleIndicator2)
        circleVerticalLayout.addWidget(elevationLabel)

        # Add to main left layout (no stretching)
        leftLayout.addLayout(circleVerticalLayout)

        # Rest of the UI setup remains unchanged...
        armImageLayout = QHBoxLayout()
        armImageLayout.addStretch()

        # Arm Image (using absolute positioning instead of layouts)
        self.arm_image_label = QLabel(self)
        
        # Load image (same as before)
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
            self.pixmap = QPixmap("Capstone/Feedback/default_image.png")
        
        # Flip and scale
        transform = QTransform().rotate(180)
        self.pixmap = self.pixmap.transformed(transform).scaled(335, 475)
        
        # Set initial position - ADJUST THESE VALUES AS NEEDED
        image_x = 500  # Move right (increase this value)
        image_y = 75  # Move up (decrease this value)
        
        self.arm_image_label.setPixmap(self.pixmap)
        self.arm_image_label.setGeometry(image_x, image_y, self.pixmap.width(), self.pixmap.height())
        self.arm_image_label.show()
        #leftLayout.addWidget(self.arm_image_label)

        #leftLayout.addLayout(armImageLayout)

        # Directional arrows (GIFs)
        self.arrow_up = QLabel(self.arm_image_label)
        self.arrow_up.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground,True)
        self.arrow_up_movie = QMovie("Capstone/Feedback/arrows/up-arrow.gif")
        self.arrow_up_movie.setScaledSize(QSize(100, 100))
        self.arrow_up.setMovie(self.arrow_up_movie)
        self.arrow_up.setVisible(False)

        self.arrow_down = QLabel(self.arm_image_label)
        self.arrow_down.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.arrow_down_movie = QMovie("Capstone/Feedback/arrows/down-arrow.gif")
        self.arrow_down_movie.setScaledSize(QSize(100, 100))
        self.arrow_down.setMovie(self.arrow_down_movie)
        self.arrow_down.setVisible(False)

        self.arrow_left = QLabel(self.arm_image_label)
        self.arrow_left.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground,True)
        self.arrow_left_movie = QMovie("Capstone/Feedback/arrows/left-arrow.gif")
        self.arrow_left_movie.setScaledSize(QSize(100, 100))
        self.arrow_left.setMovie(self.arrow_left_movie)
        self.arrow_left.setVisible(False)

        self.arrow_right = QLabel(self.arm_image_label)
        self.arrow_right.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground,True)
        self.arrow_right_movie = QMovie("Capstone/Feedback/arrows/right-arrow.gif")
        self.arrow_right_movie.setScaledSize(QSize(100, 100))
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
        """
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
        """
        # Guided Prompts and Warnings
        rightLayout = QVBoxLayout()
        rightLayout.setContentsMargins(0, 0, 0, 0)  # Remove all margins
        rightLayout.setSpacing(0)  # Remove spacing between widgets

        # Create a container widget for the label and text edit
        prompts_container = QWidget()
        prompts_container.setLayout(QVBoxLayout())
        prompts_container.layout().setContentsMargins(0, 0, 0, 0)
        prompts_container.layout().setSpacing(0)

        # Add the label (stick to top)
        title_label = QLabel("Guided Prompts and Warnings:")
        title_label.setContentsMargins(0, 0, 0, 0)  # Remove label margins
        prompts_container.layout().addWidget(title_label)

        # Add the text edit
        self.promptsLog = QTextEdit()
        self.promptsLog.setReadOnly(True)
        self.promptsLog.setFixedHeight(150)
        self.promptsLog.setContentsMargins(0, 0, 0, 0)  # Remove text edit margins
        prompts_container.layout().addWidget(self.promptsLog)

        # Add stretch to push everything up
        prompts_container.layout().addStretch()

        # Add container to right layout
        rightLayout.addWidget(prompts_container)

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
                margin-top: 10px;  /* Add some spacing */
            }
            QPushButton:hover {
                background-color: white;
                color: black;
            }
        """)

        self.endSimulationButton.clicked.connect(self._endSimulation)
        rightLayout.addWidget(self.endSimulationButton, alignment=Qt.AlignmentFlag.AlignCenter)

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
        # Create container widget for the plot
        #plot_container = QWidget()
        #plot_container.setLayout(QVBoxLayout())
        #plot_container.layout().setContentsMargins(0, 0, 0, 0)

        print("Plotting Veins")
        
        
        # Create a matplotlib figure with a fully transparent background
        self.figure = Figure(facecolor='none')  # Transparent figure
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: transparent;")  # Transparent canvas

        # Set a fixed size for the canvas to make the plot smaller
        self.canvas.setFixedSize(500, 500)  # Adjust the size as needed

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

        self.ax.view_init(elev=self.original_elev, azim=self.original_azim)
        self._showViewLabel("Side View")

        self.view_toggle_state = False  # Start with YZ view

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

        legend = self.ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.8, 1),
            title="Legend",
            facecolor='none',         # Transparent background
            edgecolor='none',         # No border
            labelcolor='white',       # Legend text color (for Matplotlib >=3.6)
        )

        # Manually style text for broader compatibility
        for text in legend.get_texts():
            text.set_color("white")

        # Style the title
        legend.get_title().set_color("white")

        # Style the frame
        legend.get_frame().set_alpha(0)  # Fully transparent
        legend.get_frame().set_linewidth(0)

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
        self._showViewLabel("Top View")  # or "Side View" depending on the initial view

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

    def cleanup(self):
        for timer in [self.timer, self.session_timer, self.view_toggle_timer, self.live_update_timer]:
            if timer:
                timer.stop()
                timer.deleteLater()

        for optional in ["view_animation_timer", "label_fade_in_timer", "label_fade_out_timer", "rotation_timer"]:
            if hasattr(self, optional):
                getattr(self, optional).stop()
                getattr(self, optional).deleteLater()

        for movie in [self.arrow_up_movie, self.arrow_down_movie, self.arrow_left_movie, self.arrow_right_movie]:
            movie.stop()
            del movie

        self.canvas.setParent(None)
        self.ax.cla()
        self.figure.clear()
        plt.close(self.figure)

        if hasattr(self, 'live_trajectory_queue'):
            self.live_trajectory_queue.queue.clear()

        if hasattr(self, 'live_trajectory_buffer'):
            self.live_trajectory_buffer.clear()

        self.close
        self.deleteLater()

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
        
        feedback = self._generateFeedback(direction)
        
        x, y, z, pitch, roll, yaw = data
        self.live_trajectory_queue.put([x, y, z])

        simulated_position = f"({x:.2f}, {y:.2f}, {z:.2f})"
        simulated_elevation = pitch
        simulated_angle_of_insertion = yaw

        #self.positionInput.setText(simulated_position)
        #self.angleInput.setText(f"{simulated_elevation:.2f}")
        #self.depthInput.setText(f"{simulated_angle_of_insertion:.2f}")

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

    def _generateFeedback(self, direction):
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
        self.app_to_signal_processing.put([None, None])
        self.timer.stop()
        self._showSummaryPage()

    def _showSummaryPage(self):
        """Display the simulation summary with a full-screen dialog and styled 3D plot."""
        angle_error = self.total_angle_deviation / self.max_updates
        depth_error = self.total_depth_deviation / self.max_updates

        print("Getting score\n")
        score = self.user_score_queue.get()
        print("Got score\n")

        if score == 3:
            feedback = ("Excellent performance! Your actions demonstrate a high level of accuracy and precision." 
                    " Maintain this level of focus and attention to detail in future tasks. Great job!")
        elif score == 2:
            feedback = ("Good performance! You show a solid understanding of the task, but there are occasional minor errors."
                        " To improve further, double-check your movements or decisions to ensure consistency."
                        " Consider reviewing any specific steps where you felt less confident.")
        elif score == 1:
            feedback = (
                "Needs improvement. It seems there were significant challenges in accuracy or approach."
                " Take time to revisit the fundamental concepts and techniques."
                " Break the task into smaller steps, practice each one thoroughly, and don't hesitate to ask for guidance."
            )
        else:
            feedback = "No data to compare"

        # Load data
        live_traj = np.loadtxt("Capstone/Filter/filtered_data.txt")
        expert_traj = np.loadtxt("Capstone/SignalProcessing/expert_data/right-vein/middle/mean_traj.txt")

        # Create Matplotlib figure with black background
        fig = Figure(figsize=(8, 6), facecolor='black')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        # Set plot and axes styles
        ax.grid(False)
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.title.set_color('white')

        # Plot expert and live trajectories with bright colors
        ax.plot(expert_traj[:, 0], expert_traj[:, 1], expert_traj[:, 2], color='cyan', linewidth=2, label="Expert (Mean)")
        try:
            ax.plot(live_traj[:, 0], live_traj[:, 1], live_traj[:, 2], color='lime', linewidth=2, label="Live")

            # Markers
            ax.scatter(live_traj[0][0], live_traj[0][1], live_traj[0][2], color='lime', marker='o', s=50, label='Live Start')
            ax.scatter(live_traj[-1][0], live_traj[-1][1], live_traj[-1][2], color='lime', marker='x', s=50, label='Live End')
        except IndexError as e:
            pass
        ax.scatter(expert_traj[0][0], expert_traj[0][1], expert_traj[0][2], color='cyan', marker='o', s=50, label='Expert Start')
        ax.scatter(expert_traj[-1][0], expert_traj[-1][1], expert_traj[-1][2], color='cyan', marker='x', s=50, label='Expert End')


        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Live vs Expert Trajectory')

        ax.legend(facecolor='black', labelcolor='white')

        # Start rotating the 3D view using QTimer
        self.rotation_angle = 0

        def rotate_plot():
            self.rotation_angle = (self.rotation_angle + 1) % 360
            ax.view_init(elev=30, azim=self.rotation_angle)
            canvas.draw_idle()

        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(rotate_plot)
        self.rotation_timer.start(50)  # 50 ms → ~20 FPS smoothness

        # Create fullscreen dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Simulation Summary")
        dialog_layout = QVBoxLayout()

        summary_label = QLabel(f"Final Score: {score}")
        summary_label.setStyleSheet("color: white; font-size: 58px; font-weight: bold;")
        summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        feedback_label = QLabel(feedback)
        feedback_label.setWordWrap(True)
        feedback_label.setStyleSheet("color: white; font-size: 40px; font-weight: bold;")
        feedback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Return button
        returnToIntroButton = QPushButton("Return to Start")
        returnToIntroButton.clicked.connect(lambda: self._returnToIntroScreen(dialog))
        returnToIntroButton.setStyleSheet("font-size: 16px; padding: 6px;")

        # Assemble layout
        dialog_layout.addWidget(summary_label)
        dialog_layout.addWidget(feedback_label)
        dialog_layout.addWidget(canvas)
        dialog_layout.addWidget(returnToIntroButton, alignment=Qt.AlignmentFlag.AlignCenter)

        dialog.setLayout(dialog_layout)
        dialog.setStyleSheet("background-color: black;")
        dialog.showFullScreen()  # Fullscreen

        dialog.exec()


    def _returnToIntroScreen(self, dialog):
        """Return to the IntroScreen and restart the application flow."""
        dialog.close()  # Close the summary dialog

        # Stop everything
        self.cleanup()

        self.close()
        self.deleteLater()

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
                                                  direction_intruction_queue=self.direction_intruction_queue,
                                                  user_score_queue=self.user_score_queue)
                    self.feedback_ui.show()

class MainApplication:
    def __init__(self,
                 sig_processed_queue,
                 app_to_signal_processing,
                 angle_range_queue,
                 direction_intruction_queue,
                 user_score_queue):
        self.sig_processed_queue = sig_processed_queue
        self.app_to_signal_processing = app_to_signal_processing
        self.angle_range_queue = angle_range_queue
        self.direction_intruction_queue = direction_intruction_queue
        self.user_score_queue = user_score_queue
        self.app = QApplication(sys.argv)

    def run(self):
        while True:  # Loop to handle navigation
            # Intro Screen
            intro_screen = IntroScreen()
            if intro_screen.exec() == QDialog.DialogCode.Accepted:
                # Pick Vein Screen
                vein_screen = PickVeinScreen()
                result = vein_screen.exec()
                
                if result == QDialog.DialogCode.Accepted:
                    selected_vein = vein_screen.selected_vein
                    # Pick Insertion Point Screen
                    insertion_point_screen = PickInsertionPointScreen(selected_vein)
                    if insertion_point_screen.exec() == QDialog.DialogCode.Accepted:
                        selected_point = insertion_point_screen.selected_point
                        self.app_to_signal_processing.put([selected_vein, selected_point])
                        # Launch Feedback UI
                        feedback_ui = FeedbackUI(
                            selected_vein,
                            selected_point,
                            work_queue=self.sig_processed_queue,
                            angle_range_queue=self.angle_range_queue,
                            app_to_signal_processing=self.app_to_signal_processing,
                            direction_intruction_queue=self.direction_intruction_queue,
                            user_score_queue=self.user_score_queue
                        )
                        feedback_ui.show()
                        sys.exit(self.app.exec())
                else:
                    continue  # Go back to IntroScreen
            else:
                break  # Exit the app