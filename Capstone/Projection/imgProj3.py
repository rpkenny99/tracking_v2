import sys
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join("Capstone", "Feedback"))
from feedback3 import FeedbackUI  # Import UI
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QPixmap
from scipy.spatial.transform import Rotation as R



class ProjectionAdjustment:
    def __init__(self, position_queue):

        self.filler_matrix = np.array([
            [1, 0, 0],  # No scaling or skewing in X
            [0, 1, 0],  # No scaling or skewing in Y
            [0, 0, 1]   # Homogeneous coordinate
        ], dtype=np.float32)
        self.position_queue = position_queue  # Queue with position/orientation data

        # Initialize the UI
        self.app = QApplication(sys.argv)
        self.ui = FeedbackUI("Left Vein", "Point A")  # Example, update dynamically

        # Transformation parameters (to be updated dynamically)
        self.translation = np.array([0, 0, 0])  # (x, y, z)
        self.rotation = np.array([0, 0, 0])  # (yaw, pitch, roll)

        self.update_display()

    def get_transform_matrix(self):
        """Compute the transformation matrix from position and orientation."""
        # Convert rotation vector to rotation matrix
        r_matrix = R.from_euler('xyz', self.rotation, degrees=True).as_matrix()

        # Homogeneous transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = r_matrix  # Apply rotation
        transform_matrix[:3, 3] = self.translation  # Apply translation
        return transform_matrix

    def apply_projection_adjustment(self):
        """Apply the transformation to UI elements."""
        # Get the latest position/orientation data
        if not self.position_queue.empty():
            data = self.position_queue.get()  # Example: (x, y, z, yaw, pitch, roll)
            self.translation = np.array(data[:3])
            self.rotation = np.array(data[3:])

        # Compute transformation matrix
        transform_matrix = self.get_transform_matrix()

        # Update UI display (example: apply transformation to an overlay image)
        new_pixmap = self.transform_image(self.ui.arm_image_label.pixmap(), transform_matrix)
        # self.filler_matrix
        # self.transform_image(self.ui.arm_image_label.pixmap(), transform_matrix)
        self.ui.arm_image_label.setPixmap(new_pixmap)

    def transform_image(self, pixmap, transform_matrix):
        """Apply a perspective warp to the UI elements."""
        img = pixmap.toImage()

        # Convert QImage to NumPy array using correct format
        img = img.convertToFormat(QImage.Format.Format_ARGB32)  # Correct for PyQt6
        width, height = img.width(), img.height()

        # Convert QImage to NumPy array
        ptr = img.bits()
        ptr.setsize(height * width * 4)  # Ensure correct buffer size
        img_array = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))  # Explicit dtype

        # Define original image points
        src_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        # Apply transformation to get new projected points
        dst_pts = cv2.perspectiveTransform(np.array([src_pts]), transform_matrix[:3, :3])[0]

        # Compute homography
        h_matrix, _ = cv2.findHomography(src_pts, dst_pts)

        # Apply transformation using OpenCV
        warped_img = cv2.warpPerspective(img_array, h_matrix, (width, height))

        # Convert back to QPixmap
        warped_qimage = QImage(warped_img.data, width, height, QImage.Format.Format_ARGB32)
        return QPixmap.fromImage(warped_qimage)




    def update_display(self):
        """Continuously update display based on user position."""
        while True:
            self.apply_projection_adjustment()
            self.ui.repaint()

if __name__ == "__main__":
    from queue import Queue
    position_queue = Queue()

    # Example: simulate position/orientation updates
    position_queue.put((0.1, 0.2, -0.1, 10, -5, 3))  # (x, y, z, yaw, pitch, roll)

    projection_adjustment = ProjectionAdjustment(position_queue)
    sys.exit(projection_adjustment.app.exec())
