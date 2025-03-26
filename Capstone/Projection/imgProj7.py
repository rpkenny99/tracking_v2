import sys
import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from feedback3 import FeedbackUI  # Import your UI class
from queue import Queue

class ProjectionWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)  # Ensure the pixmap scales to the widget size

    # Override mouse event handlers to capture interactivity
    def mousePressEvent(self, event):
        print("Mouse pressed at:", event.position().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        print("Mouse moved at:", event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        print("Mouse released at:", event.position().toPoint())
        super().mouseReleaseEvent(event)

def qpixmap_to_numpy(pixmap):
    """
    Convert a QPixmap to a NumPy array (BGR format for OpenCV).
    """
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

def apply_projection_transform(image, angle_deg=25):
    """
    Applies a perspective transformation to simulate a projection plane
    rotated by angle_deg relative to the source screen.
    """
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle_deg)
    new_top_width = w * np.cos(angle_rad)
    offset = (w - new_top_width) / 2

    src_pts = np.float32([[0, 0],
                          [w, 0],
                          [w, h],
                          [0, h]])
    dst_pts = np.float32([[offset, 0],
                          [w - offset, 0],
                          [w, h],
                          [0, h]])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def update_display(feedback_ui, projection_widget, angle_deg=25):
    """
    Capture the updated UI, flip it for mirror reflection, apply the projection transform,
    and then update the projection widget with the new image.
    """
    # Capture the current UI as a QPixmap
    pixmap = feedback_ui.grab()
    # Convert QPixmap to a NumPy array in BGR format
    original_image = qpixmap_to_numpy(pixmap)
    
    # Optionally, resize the image if its dimensions are unexpectedly large.
    h, w = original_image.shape[:2]
    max_dim = 3000  # You can adjust this value if needed.
    if w > max_dim or h > max_dim:
        scale = max_dim / float(max(w, h))
        original_image = cv2.resize(original_image, (int(w * scale), int(h * scale)))
        h, w = original_image.shape[:2]
    
    # Flip the image (simulate mirror reflection)
    flipped_image = cv2.flip(original_image, 0)
    # Apply the perspective transformation
    transformed_image = apply_projection_transform(flipped_image, angle_deg=angle_deg)
    
    # Convert the transformed image from BGR to RGB for display in Qt
    transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    h, w, ch = transformed_image_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(transformed_image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    
    # Update the projection widget with the new QPixmap
    projection_widget.setPixmap(QPixmap.fromImage(qimg))

def start(sig_processed):
    app = QApplication(sys.argv)

    # Supply required parameters and a queue for FeedbackUI
    selected_vein = "Left Vein"
    selected_point = "Point A"
    feedback_ui = FeedbackUI(selected_vein, selected_point, work_queue=sig_processed)
    
    # Create an instance of our interactive projection widget
    projection_widget = ProjectionWidget()

    # Create a main container widget to display both the FeedbackUI and the ProjectionWidget side by side
    main_widget = QWidget()
    layout = QHBoxLayout()
    layout.addWidget(feedback_ui)
    layout.addWidget(projection_widget)
    main_widget.setLayout(layout)
    main_widget.show()

    # Set up a QTimer to update the projection display periodically (every 100 ms)
    timer = QTimer()
    timer.timeout.connect(lambda: update_display(feedback_ui, projection_widget, angle_deg=25))
    timer.start(100)

    sys.exit(app.exec())

if __name__ == "__main__":
    start(sig_processed=Queue())
