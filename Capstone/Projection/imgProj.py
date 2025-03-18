import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from PyQt6.QtWidgets import QApplication, QDialog
# import feedback3  # Importing your UI module
import sys
# sys.path.append("Capstone/Projection")
import os
sys.path.append(os.path.join("Capstone", "Feedback"))
import feedback3
import queue
import threading

class ProjectionAlignment:
    def __init__(self, ui_instance, position_source):
        """Initializes projection alignment with a UI instance and a data source."""
        self.ui = ui_instance  # Reference to the FeedbackUI
        self.position_source = position_source  # Function to get position and orientation

    # def get_transformation_matrix(self, position, orientation):
    #     """Compute the transformation matrix from user's perspective."""
    #     rotation_matrix = R.from_euler('xyz', orientation, degrees=True).as_matrix()
    #     translation_vector = np.array(position).reshape(3, 1)
        
    #     transformation_matrix = np.eye(4)
    #     transformation_matrix[:3, :3] = rotation_matrix
    #     transformation_matrix[:3, 3] = translation_vector.ravel()
        
    #     return transformation_matrix

    def get_transformation_matrix(self, rVec, tVec, ref_tVec):
        """Compute transformation matrix using a reference point for alignment."""
        
        # Convert Rodrigues rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rVec)

        # Ensure translation vector is a (3,1) column vector
        translation_vector = np.array(tVec).reshape(3, 1)

        # Create the initial transformation matrix (before adjustment)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix  # Insert rotation
        transformation_matrix[:3, 3] = translation_vector.ravel()  # Insert translation

        # Compute offset based on the reference point
        ref_offset = np.array(ref_tVec).reshape(3) - np.array(tVec).reshape(3)

        # Apply reference point correction to align the projection
        transformation_matrix[:3, 3] += ref_offset

        return transformation_matrix


    def update_projection(self):
        """Updates the projection based on the user's position and orientation."""
        position, orientation = self.position_source()  # Get real-time data
        transformation_matrix = self.get_transformation_matrix(position, orientation)
        
        # Apply transformation to UI elements (adjusting displayed overlay accordingly)
        self.ui.apply_transformation(transformation_matrix)

# Example function to retrieve real-time position and orientation
def mock_position_source():
    """Returns dummy position and orientation data."""
    position = [0.5, 0.2, -1.0]  # Example XYZ position
    orientation = [10, -5, 3]  # Example XYZ rotation in degrees
    return position, orientation

# def phantom_position_source():
#     """Returns dummy position and orientation data."""
#     position = [0.5, 0.2, -1.0]  # Example XYZ position
#     orientation = [10, -5, 3]  # Example XYZ rotation in degrees
#     return position, orientation

# if __name__ == "__main__":
#     queue = feedback3.Queue()
#     ui_instance = feedback3.FeedbackUI("Left Vein", "Point A", work_queue=queue)
#     projection_alignment = ProjectionAlignment(ui_instance, mock_position_source)
    
#     # Continuously update projection in real-time
#     while True:
#         projection_alignment.update_projection()


if __name__ == "__main__":
    # Step 1: Initialize QApplication
    app = QApplication(sys.argv)

    # Step 2: Show Intro Screen
    intro_screen = feedback3.IntroScreen()
    if intro_screen.exec() != QDialog.DialogCode.Accepted:
        sys.exit()  # Exit if user doesn't start simulation

    # Step 3: Show Vein Selection Screen
    vein_screen = feedback3.PickVeinScreen()
    if vein_screen.exec() != QDialog.DialogCode.Accepted:
        sys.exit()  # Exit if user cancels

    selected_vein = vein_screen.selected_vein
    print(f"Selected Vein: {selected_vein}")  # Debugging output

    # Step 4: Show Insertion Point Selection Screen
    insertion_screen = feedback3.PickInsertionPointScreen(selected_vein)
    if insertion_screen.exec() != QDialog.DialogCode.Accepted:
        sys.exit()  # Exit if user cancels

    selected_point = insertion_screen.selected_point
    print(f"Selected Insertion Point: {selected_point}")  # Debugging output

    # Step 5: Launch Feedback UI
    queue = feedback3.Queue()  # Initialize queue for processing
    ui_instance = feedback3.FeedbackUI(selected_vein, selected_point, work_queue=queue)
    ui_instance.show()  # Show the UI window

    # Step 6: Initialize Projection Alignment (with mock or real position source)
    projection_alignment = feedback3.ProjectionAlignment(ui_instance, feedback3.mock_position_source)

    # Step 7: Start PyQt event loop
    sys.exit(app.exec())  # Runs the application

