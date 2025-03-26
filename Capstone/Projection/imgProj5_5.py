import sys
import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
from feedback3 import FeedbackUI  # Import your UI class
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

def apply_projection_transform(image, angle_deg=-25):
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

def update_display(feedback_ui, angle_deg=-25, scale_factor=0.7, offset_x=0, offset_y=0):
    """
    Capture the updated UI, flip it for reflection, apply the projection transform,
    resize the result to occupy less space, and then display it with an offset.
    """
    # Capture the current UI as a QPixmap
    pixmap = feedback_ui.grab()
    # Convert to a NumPy array (BGR format)
    original_image = qpixmap_to_numpy(pixmap)
    
    # Flip horizontally if the projected image is reflected
    flipped_image = cv2.flip(original_image, 0)
    
    # Apply projection transformation
    transformed_image = apply_projection_transform(flipped_image, angle_deg=angle_deg)

    # Get original dimensions
    window_height, window_width = transformed_image.shape[:2]

    # Scale down the image
    new_width = int(window_width * scale_factor)
    new_height = int(window_height * scale_factor)
    resized_image = cv2.resize(transformed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create a blank image (same size as the original window) with black background
    display_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)

    # Calculate the position to center the resized image and then add the offsets
    start_x = (window_width - new_width) // 2 + offset_x
    start_y = (window_height - new_height) // 2 + offset_y

    # Overlay the resized image onto the blank canvas
    display_image[start_y:start_y+new_height, start_x:start_x+new_width] = resized_image

    # Display the result using OpenCV
    cv2.imshow("Projected UI", display_image)
    cv2.waitKey(1)  # Small delay for OpenCV event processing

def start(sig_processed):
    app = QApplication(sys.argv)

    # Supply required parameters and a dummy queue for FeedbackUI
    selected_vein = "Left Vein"
    selected_point = "Point A"
    work_queue = Queue()  # Provide an empty queue if needed

    feedback_ui = FeedbackUI(selected_vein, selected_point, work_queue=sig_processed)
    feedback_ui.show()

    # Set up a QTimer to update the projection capture periodically (every 100 ms)
    # Adjust offset_x and offset_y here to displace the image from the center.
    timer = QTimer()
    timer.timeout.connect(lambda: update_display(feedback_ui, angle_deg=25, offset_x=100, offset_y=50))
    timer.start(100)  # Update every 100 milliseconds

    sys.exit(app.exec())

if __name__ == "__main__":
    start(sig_processed=Queue())
