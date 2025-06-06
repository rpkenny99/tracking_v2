import sys
import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsProxyWidget, QGraphicsView
from PyQt6.QtGui import QImage, QTransform
from Feedback.feedback3 import FeedbackUI  # Import your UI class
from queue import Queue

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

def update_display(feedback_ui, angle_deg=-25):
    """
    Capture the updated UI, flip it for reflection, apply the projection transform,
    and then display the result.
    """
    # Capture the current UI as a QPixmap
    pixmap = feedback_ui.grab()
    # Convert to a NumPy array (BGR format)
    original_image = qpixmap_to_numpy(pixmap)
    
    # Flip horizontally if the projected image is reflected
    flipped_image = cv2.flip(original_image, 0)
    
    # Apply projection transformation
    transformed_image = apply_projection_transform(flipped_image, angle_deg=angle_deg)
    
    # Display the result using OpenCV
    cv2.imshow("Projected UI", transformed_image)
    cv2.waitKey(1)  # Small delay for OpenCV event processing

# def start(sig_processed):
#     app = QApplication(sys.argv)

#     # Supply required parameters and a dummy queue for FeedbackUI
#     selected_vein = None
#     selected_point = None
#     work_queue = Queue()  # Provide an empty queue if needed

#     feedback_ui = FeedbackUI(selected_vein, selected_point, work_queue=sig_processed)
#     feedback_ui.show()

#     # Set up a QTimer to update the projection capture periodically (every 100 ms)
#     timer = QTimer()
#     timer.timeout.connect(lambda: update_display(feedback_ui, angle_deg=25))
#     timer.start(100)  # Update every 100 milliseconds

#     sys.exit(app.exec())

def start(sig_processed):
    app = QApplication(sys.argv)

    # Create your interactive UI with required parameters.
    selected_vein = "Left Vein"
    selected_point = "Point A"
    # feedback_ui = FeedbackUI(selected_vein, selected_point)
    feedback_ui = FeedbackUI(selected_vein, selected_point, work_queue=sig_processed)

    
    # Create a QGraphicsScene and embed the widget as a proxy.
    scene = QGraphicsScene()
    proxy = QGraphicsProxyWidget()
    proxy.setWidget(feedback_ui)
    scene.addItem(proxy)
    
    # Create a QGraphicsView to display the scene.
    view = QGraphicsView(scene)
    
    # Apply a perspective-like transformation.
    # For example, this transformation scales the top edge:
    angle_deg = 25
    scale_factor = 1.0  # adjust as needed for your effect
    transform = QTransform()
    # Note: You might need a custom QTransform for a true perspective effect.
    # Here we simulate it with a shear transformation:
    shear_factor = 0.2  # tweak this factor to simulate tilt
    transform.shear(0, shear_factor)
    proxy.setTransform(transform)
    
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    start()
