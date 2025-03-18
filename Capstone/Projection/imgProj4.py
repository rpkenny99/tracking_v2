import sys
import numpy as np
import cv2
import os
from queue import Queue, Empty
from PyQt6.QtWidgets import QApplication, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from scipy.spatial.transform import Rotation as R

# sys.path.append(os.path.join("Capstone", "Feedback", "feedback3"))
sys.path.append("/2025Capstone_Env/tracking_v2/Capstone/Feedback")

# Import the module correctly
from feedback3 import FeedbackUI  # Import UI

class ProjectionAdjustment:
    def __init__(self, position_queue):
        self.position_queue = position_queue  # Queue with position/orientation data

        # Initialize the UI
        self.app = QApplication(sys.argv)
        self.ui = FeedbackUI("Left Vein", "Point A")  # Example, update dynamically

        # Transformation parameters (to be updated dynamically)
        self.translation = np.array([0, 0, 0])  # (x, y, z)
        self.rotation = np.array([0, 0, 0])  # (yaw, pitch, roll)

        # Timer to update the display without blocking the UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.apply_projection_adjustment)
        self.timer.start(100)  # Update every 100ms
    
    def get_transform_matrix(self):
        """Compute the transformation matrix from position and orientation."""
        r_matrix = R.from_euler('xyz', self.rotation, degrees=True).as_matrix()
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = r_matrix  # Apply rotation
        transform_matrix[:3, 3] = self.translation  # Apply translation
        return transform_matrix

    def apply_projection_adjustment(self):
        """Apply the transformation to UI elements."""
        try:
            data = self.position_queue.get_nowait()  # Non-blocking
            self.translation = np.array(data[:3])
            self.rotation = np.array(data[3:])
        except Empty:
            return

        transform_matrix = self.get_transform_matrix()
        new_pixmap = self.transform_image(self.ui.arm_image_label.pixmap(), transform_matrix)
        self.ui.arm_image_label.setPixmap(new_pixmap)

    def transform_image(self, pixmap, transform_matrix):
        """Apply a perspective warp to the UI elements."""
        img = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        width, height = img.width(), img.height()

        # Convert QImage to NumPy array
        ptr = img.bits()
        ptr.setsize(height * width * 4)
        img_array = np.frombuffer(ptr.asstring(width * height * 4), dtype=np.uint8).reshape((height, width, 4))

        # Define original image points
        src_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]]).reshape(-1, 1, 2)
        
        # Apply transformation
        dst_pts = cv2.perspectiveTransform(src_pts, transform_matrix[:3, :3])
        if dst_pts is None or dst_pts.shape != (4, 1, 2):
            return pixmap  # Return original if transformation fails
        
        h_matrix, _ = cv2.findHomography(src_pts, dst_pts)
        if h_matrix is None:
            return pixmap  # Return original if homography computation fails

        warped_img = cv2.warpPerspective(img_array, h_matrix, (width, height))
        warped_qimage = QImage(warped_img.data, width, height, QImage.Format_ARGB32)
        return QPixmap.fromImage(warped_qimage)

if __name__ == "__main__":
    position_queue = Queue()
    position_queue.put((0.1, 0.2, -0.1, 10, -5, 3))  # Simulated data
    projection_adjustment = ProjectionAdjustment(position_queue)
    sys.exit(projection_adjustment.app.exec())
