import sys
import cv2
import numpy as np
import math
from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QApplication
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage
from updated_feeback import FeedbackUI  # Import your UI from the shared file
from queue import Queue


class TransformedDisplay(QMainWindow):
    def __init__(self, original_widget):
        super().__init__()
        self.setWindowTitle("Transformed Display (25° Tilt)")
        self.original_widget = original_widget  # This is our source display (FeedbackUI)
        
        # Create a QLabel to show the transformed image.
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.label)
        
        # Grab an initial frame to determine dimensions.
        pixmap = self.original_widget.grab()
        self.frame_width = pixmap.width()
        self.frame_height = pixmap.height()
        
        # Define source points (the original corners).
        src_points = np.float32([
            [0, 0],
            [self.frame_width - 1, 0],
            [self.frame_width - 1, self.frame_height - 1],
            [0, self.frame_height - 1]
        ])
        
        # Compute offset for a 25° tilt (using tan(angle)).
        angle_deg = 25.0
        angle_rad = math.radians(angle_deg)
        offset = self.frame_height * math.tan(angle_rad)
        
        # Define destination points to simulate the tilt.
        dst_points = np.float32([
            [0, 0],
            [self.frame_width - 1, 0],
            [self.frame_width - 1 + offset, self.frame_height - 1],
            [0 + offset, self.frame_height - 1]
        ])
        
        # Compute the transformation matrix and its inverse.
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        # Set up a timer to update the transformed display.
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(30)  # Update roughly every 30 ms (~33 fps)
        
    def update_display(self):
        # Capture the current content from the original widget.
        pixmap = self.original_widget.grab()
        
        # Convert QPixmap to QImage.
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        width = image.width()
        height = image.height()
        
        # Access the image bits and create a NumPy array.
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)
        
        # Apply the perspective transformation using OpenCV.
        transformed_arr = cv2.warpPerspective(arr, self.transform_matrix, (self.frame_width, self.frame_height))
        
        # Convert the transformed array back to QImage.
        transformed_qimage = QImage(
            transformed_arr.data,
            transformed_arr.shape[1],
            transformed_arr.shape[0],
            transformed_arr.strides[0],
            QImage.Format.Format_RGB888
        )
        transformed_pixmap = QPixmap.fromImage(transformed_qimage)
        
        # Set the transformed image to our label.
        self.label.setPixmap(transformed_pixmap)
        
    def mousePressEvent(self, event):
        # When a user clicks on the transformed display, map that click back to the original coordinates.
        x = event.position().x()
        y = event.position().y()
        # Prepare the point in the expected shape for cv2.perspectiveTransform.
        pt = np.array([[[x, y]]], dtype=np.float32)
        original_pt = cv2.perspectiveTransform(pt, self.inverse_transform_matrix)
        original_x, original_y = original_pt[0][0]
        print(f"Clicked on transformed display at ({x:.1f}, {y:.1f}) -> mapped to original: ({original_x:.1f}, {original_y:.1f})")
        # Here you can add further interactivity based on original coordinates.

def main():
    # Create dummy queues for initializing the FeedbackUI.
    work_queue = Queue()
    angle_range_queue = Queue()
    app_to_signal_processing = Queue()
    # Push dummy angle and standard deviation values as required.
    angle_range_queue.put((30, 0, 0, 5, 0, 5))
    
    app = QApplication(sys.argv)
    
    # Instantiate the original FeedbackUI from your updated_feeback module.
    # Adjust the parameters ("Left Vein", "Point A") as needed for your use case.
    feedback_ui = FeedbackUI("Left Vein", "Point A", work_queue, angle_range_queue, app_to_signal_processing)
    feedback_ui.show()
    
    # Create the transformed display that captures from the feedback_ui.
    transformed_display = TransformedDisplay(feedback_ui)
    transformed_display.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
