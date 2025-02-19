from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QTextEdit, QGroupBox, QHBoxLayout, QDialog,QStackedWidget
)
from PyQt6.QtGui import QFont
import sys
from random import uniform

class IntroScreen(QDialog):
    """Introductory screen to start the simulation."""
    def __init__(self,):
        super().__init__()
        self.setWindowTitle("Cyber-Physical Infant IV Simulator")
        self.setFixedSize(1365, 680)

        layout = QVBoxLayout()
        label = QLabel("Cyber-Physical Infant IV Simulator")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        font = QFont()
        font.setPointSize(26)
        font.setBold(True)
        label.setFont(font)

        layout.addWidget(label)

        start_button = QPushButton("Start Simulation")
        start_button.clicked.connect(self.accept) #close the dialog on click
        layout.addWidget(start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)
    
    #def start_simulation(self):
        #self.done(QDialog.DialogCode.Accepted)

class PickVeinScreen(QDialog):
    "Choose the Vein to be pierced"
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pick Your Vein")
        self.setFixedSize(1365, 680)

        layout = QVBoxLayout()
            
        #Header
        header_label = QLabel("Pick Your Vein")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        header_label.setFont(font)
        layout.addWidget(header_label)


        # Arm Image
        arm_image_label = QLabel(self)
        pixmap = QPixmap("aarm.png")
        arm_image_label.setPixmap(pixmap.scaled(600, 300,Qt.AspectRatioMode.KeepAspectRatio))
        arm_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(arm_image_label)

        #Buttons for vein selection
        button_layout = QHBoxLayout()
        left_vein_button = QPushButton("Left Vein")
        left_vein_button.clicked.connect(self.select_left_vein)
        button_layout.addWidget(left_vein_button)

        right_vein_button = QPushButton("Right Vein")
        right_vein_button.clicked.connect(self.select_right_vein)
        button_layout.addWidget(right_vein_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def select_left_vein(self):
    #Handle selection of the left vein
        self.selected_vein = "Left vein"
        self.accept()

    def select_right_vein(self):
    #Handle selection of the right vein
        self.selected_vein = "Right Vein"
        self.accept()

class PickInsertionPointScreen(QDialog):
    #Screen to chose the insertion point
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pick Insertion Point")
        self.setFixedSize(1365, 680)

        layout = QVBoxLayout()

        #Header    
        header_label = QLabel("Pick Insertion Point")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        header_label.setFont(font)
        layout.addWidget(header_label)

        #Arm Image with Clickable Points
        arm_image_label = QLabel(self)
        pixmap = QPixmap("pickinsertionpoint.png")
        arm_image_label.setPixmap(pixmap.scaled(600, 300, Qt.AspectRatioMode.KeepAspectRatio))
        arm_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(arm_image_label)

        #Buttons for insertion points
        button_layout = QHBoxLayout()
        point_a_button = QPushButton("Point A")
        point_a_button.clicked.connect(self.select_point_a)
        button_layout.addWidget(point_a_button)

        point_b_button = QPushButton("Point B")
        point_b_button.clicked.connect(self.select_point_b)
        button_layout.addWidget(point_b_button)

        point_c_button = QPushButton("Point C")
        point_c_button.clicked.connect(self.select_point_c)
        button_layout.addWidget(point_c_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def select_point_a(self):
        #Handle selection of Point A
        self.selected_point = "Point A"
        self.accept()

    def select_point_b(self):
        #Handle selection of Point B
        self.selected_point = "Point B"
        self.accept()

    def select_point_c(self):
        #Handle selection of Point C
        self.selected_point = "Point C"
        self.accept()

class FeedbackUI(QMainWindow):
    def __init__(self, max_updates=7, update_interval=1000):
        super().__init__()
        self.setWindowTitle("Feedback UI - Needle Insertion")
        self.setFixedSize(1365, 680)

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

        self._createDisplay()

        # Timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._updateData)
        self.timer.start(update_interval)

    def _createDisplay(self):
        """Create the UI layout with the updated design."""
        leftLayout = QVBoxLayout()

        # Arm Image
        self.arm_image_label = QLabel(self)
        pixmap = QPixmap("transparent.png")
        self.arm_image_label.setPixmap(pixmap.scaled(335, 475))
        self.arm_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        leftLayout.addWidget(self.arm_image_label)

        self._createTargetMetricsBox(leftLayout)

        # Directional arrows (GIFs)
        self.arrow_up = QLabel(self.arm_image_label)
        self.arrow_up_movie = QMovie("arrow-1153-256-mainup-1-unscreen")
        self.arrow_up.setMovie(self.arrow_up_movie)
        #self.arrow_up.setFixedSize(50, 50)
        self.arrow_up.setVisible(False)

        self.arrow_down = QLabel(self.arm_image_label)
        self.arrow_down_movie = QMovie("arrow-1153_256(maindown).gif")
        self.arrow_down.setMovie(self.arrow_down_movie)
        #self.arrow_down.setFixedSize(50, 50)
        self.arrow_down.setVisible(False)

        self.arrow_left = QLabel(self.arm_image_label)
        self.arrow_left_movie = QMovie("arrow-358_256(mainleft).gif")
        self.arrow_left.setMovie(self.arrow_left_movie)
        #self.arrow_left.setFixedSize(50, 50)
        self.arrow_left.setVisible(False)

        self.arrow_right = QLabel(self.arm_image_label)
        self.arrow_right_movie = QMovie("arrow-358-256-mainright--unscreen")
        self.arrow_right.setMovie(self.arrow_right_movie)
        #self.arrow_right.setFixedSize(50, 50)
        self.arrow_right.setVisible(False)

        # Position Arrows (overlayed)
        self.arrow_up.move(600, 55)  # Adjust position as needed
        self.arrow_down.move(600, 310)
        self.arrow_left.move(395, 180)
        self.arrow_right.move(700, 190)

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

        # Add layouts to the general layout
        self.generalLayout.addLayout(leftLayout, 1)  # Assign more weight to the left layout
        self.generalLayout.addLayout(rightLayout)

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

    def _updateData(self):
        """Simulate data updates and display feedback for needle guidance."""
        # Stop updates if maximum count is reached
        if self.update_count >= self.max_updates:
            self.timer.stop()  # Stop the timer
            self.promptsLog.append("Simulation complete.")  # Log completion message
            self._showSummaryPage()  # Transition to summary page
            return

        # Generate random deviations to simulate position, angle, and depth
        simulated_position = f"({uniform(-5, 5):.2f}, {uniform(-5, 5):.2f}, {uniform(-5, 5):.2f})"
        simulated_angle = uniform(25, 31)  # Random angle deviation #Where we tweak the random data
        simulated_depth = uniform(40, 51)  # Random depth deviation

        # Update UI inputs
        self.positionInput.setText(simulated_position)
        self.angleInput.setText(f"{simulated_angle:.2f}")
        self.depthInput.setText(f"{simulated_depth:.2f}")

        # Generate feedback and log updates
        feedback = self._generateFeedback(simulated_position, simulated_angle, simulated_depth)
        self.promptsLog.append(
            f"Update {self.update_count + 1}: {feedback}"
        )

        # Update cumulative deviations  
        target_angle = float(self.targetAngle.text())
        target_depth = float(self.targetDepth.text())
        self.total_angle_deviation += abs(simulated_angle - target_angle)
        self.total_depth_deviation += abs(simulated_depth - target_depth)

        # Display directional arrows based on simulated data
        #self._setArrowVisibility("up", simulated_depth > 5)
        #self._setArrowVisibility("down", simulated_depth < -5)
        #self._setArrowVisibility("right", simulated_angle > 10)
        #self._setArrowVisibility("left", simulated_angle < -10)

        # Increment the update count
        self.update_count += 1


    def _setArrowVisibility(self, direction, is_visible):
        """Control visibility of directional arrows based on conditions."""
        arrow_dict = {  # Map directions to corresponding arrow labels
            "up": self.arrow_up,
            "down": self.arrow_down,
            "left": self.arrow_left,
            "right": self.arrow_right
        }
        arrow = arrow_dict.get(direction)  # Get the arrow label for the direction
        if arrow:  # If the direction is valid
            arrow.setVisible(is_visible)  # Show or hide the arrow

        
    def _generateTestData(self):
        """Generate randomized test data."""
        x, y, z = uniform(-10, 10), uniform(-10, 10), uniform(-10, 10)
        angle = uniform(0, 90)
        depth = uniform(0, 100)
        return f"({x:.2f}, {y:.2f}, {z:.2f})", angle, depth
    
    def _generateFeedback(self, position, angle, depth):
        """Generate feedback based on current and target metrics."""
        target_angle = float(self.targetAngle.text())
        target_depth = float(self.targetDepth.text())
        x = float(position.strip("()").split(",")[0])  # Extract x-coordinate

        feedback = []

        # Reset all arrows to invisible before showing the appropriate one
        self.arrow_up.setVisible(False)
        self.arrow_down.setVisible(False)
        self.arrow_left.setVisible(False)
        self.arrow_right.setVisible(False)

        # Prioritize angle feedback over position
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

        # Depth feedback is additional (doesn't affect arrow visibility)
        if depth < target_depth:
            feedback.append("Insert the needle a bit deeper.")
        elif depth > target_depth:
            feedback.append("Pull the needle out slightly.")

        return "\n".join(feedback) + "\n" + "-" * 40


    def _logPrompt(self, message):
        """Display guided prompts and warnings in the log."""
        self.promptsLog.append(message)

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
                       "Maintain this level of focus and attention to detail in future tasks. Great job!"
            )
        elif score >= 7:
            feedback = ("Good performance! You show a solid understanding of the task, but there are occasional minor errors."
                        "To improve further, double-check your movements or decisions to ensure consistency. " 
                        "Consider reviewing any specific steps where you felt less confident."          
            )
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

"""
class MainApp(QStackedWidget):
    #Main application managing the screens.
    def __init__(self):
        super().__init__()

        # Intro screen
        self.intro_screen = IntroScreen()
        self.intro_screen.accepted.connect(self.show_pick_vein_screen)
        self.addWidget(self.intro_screen)

        # Pick vein screen
        self.pick_vein_screen = PickVeinScreen()
        self.pick_vein_screen.accepted.connect(self.show_feedback_ui)
        self.addWidget(self.pick_vein_screen)

        # Feedback UI
        self.feedback_ui = FeedbackUI()
        self.addWidget(self.feedback_ui)

        self.setCurrentWidget(self.intro_screen)

    def show_pick_vein_screen(self):
        #Switch to the vein selection screen.
        self.setCurrentWidget(self.pick_vein_screen)

    def show_feedback_ui(self):
        #Switch to the feedback UI after vein selection.
        selected_vein = self.pick_vein_screen.selected_vein
        print(f"Selected Vein: {selected_vein}")  # Debugging: print selected vein
        self.setCurrentWidget(self.feedback_ui)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec())
"""


"""
class MainWindow(QMainWindow):
    #Main window with stacked screens
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cyber-Physical Infant IV Simulator")
        self.setFixedSize(900, 600)

        # Create the stacked widget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Add screens to the stacked widget
        self.intro_screen = IntroScreen()
        self.vein_screen = PickVeinScreen()
        self.feedback_ui = FeedbackUI()

        self.stacked_widget.addWidget(self.intro_screen)
        self.stacked_widget.addWidget(self.vein_screen)
        self.stacked_widget.addWidget(self.feedback_ui)

        # Connect transitions
        self.intro_screen.accepted.connect(self.show_vein_screen)
        self.vein_screen.accepted.connect(self.show_feedback_ui)

        # Start with the intro screen
        self.stacked_widget.setCurrentWidget(self.intro_screen)

    def show_vein_screen(self):
        self.stacked_widget.setCurrentWidget(self.vein_screen)

    def show_feedback_ui(self):
        self.stacked_widget.setCurrentWidget(self.feedback_ui)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
"""



class MainApplication:
    #Manages the flow of the application
    def __init__(self):
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

                # Pick Insertion Point Screen
                insertion_point_screen = PickInsertionPointScreen()
                if insertion_point_screen.exec() == QDialog.DialogCode.Accepted:
                    self.selected_insertion_point = insertion_point_screen.selected_point

                    # Launch Feedback UI
                    feedback_ui = FeedbackUI()
                    feedback_ui.show()
                    sys.exit(self.app.exec())


if __name__ == "__main__":
    main_app = MainApplication()
    main_app.run()
