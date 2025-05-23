import sys
import cv2
import numpy as np
import threading
import time
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

def compute_viewer_offset(viewer_rvec, viewer_tvec, image_width, image_height, f=500):
    """
    Computes the offset required so that the image center aligns with the
    viewer's line of sight. A simple pinhole camera model is assumed with focal length f.
    
    Parameters:
        viewer_rvec (np.array): Rotation vector (unused in this simple model)
        viewer_tvec (np.array): Translation vector (shape (1,3) or (3,))
        image_width (int): Width of the display canvas in pixels
        image_height (int): Height of the display canvas in pixels
        f (int): Assumed focal length in pixel units
        
    Returns:
        offset_x, offset_y (int): Offsets in pixels to adjust the image center.
    """
    # Extract X, Y, Z from the translation vector
    X, Y, Z = viewer_tvec.flatten()
    # Project the origin using the pinhole camera model
    proj_x = f * X / Z + image_width / 2
    proj_y = f * Y / Z + image_height / 2
    # Compute the offset needed to bring the canvas center to this projected point
    offset_x = proj_x - (image_width / 2)
    offset_y = proj_y - (image_height / 2)
    return int(offset_x), int(offset_y)

def update_display(feedback_ui, angle_deg=-25, scale_factor=0.7, offset_x=0, offset_y=0,
                   viewer_rvec=None, viewer_tvec=None):
    """
    Capture the updated UI, flip it for reflection, apply the projection transform,
    resize the result, and then display it with an offset.
    
    If viewer_rvec and viewer_tvec are provided, the offset is computed so that
    the image center aligns with the viewer's line of sight.
    """
    # Capture the current UI as a QPixmap
    pixmap = feedback_ui.grab()
    # Convert to a NumPy array (BGR format)
    original_image = qpixmap_to_numpy(pixmap)
    
    # Flip the image vertically (simulate reflection)
    flipped_image = cv2.flip(original_image, 0)
    
    # Apply the projection transformation
    transformed_image = apply_projection_transform(flipped_image, angle_deg=angle_deg)

    # Get the dimensions of the transformed image (our display canvas)
    window_height, window_width = transformed_image.shape[:2]

    # Scale down the image
    new_width = int(window_width * scale_factor)
    new_height = int(window_height * scale_factor)
    resized_image = cv2.resize(transformed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # If viewer data is provided, compute offsets based on the viewer's perspective
    if viewer_rvec is not None and viewer_tvec is not None:
        offset_x, offset_y = compute_viewer_offset(viewer_rvec, viewer_tvec, window_width, window_height)

    # Create a blank canvas with a black background
    display_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)

    # Calculate the starting position (centered plus offsets)
    start_x = (window_width - new_width) // 2 + offset_x
    start_y = (window_height - new_height) // 2 + offset_y

    # Clip the region if the resized image extends beyond the canvas
    # Determine the ROI in the canvas
    x1 = max(start_x, 0)
    y1 = max(start_y, 0)
    x2 = min(start_x + new_width, window_width)
    y2 = min(start_y + new_height, window_height)
    
    # Determine the corresponding ROI in the resized_image
    roi_x1 = 0 if start_x >= 0 else -start_x
    roi_y1 = 0 if start_y >= 0 else -start_y
    roi_x2 = roi_x1 + (x2 - x1)
    roi_y2 = roi_y1 + (y2 - y1)

    # Overlay the valid region of the resized image onto the canvas
    display_image[y1:y2, x1:x2] = resized_image[roi_y1:roi_y2, roi_x1:roi_x2]

    # Display the result using OpenCV
    cv2.imshow("Projected UI", display_image)
    cv2.waitKey(1)  # Brief delay for OpenCV event processing

def start(sig_processed):
    app = QApplication(sys.argv)

    # Supply required parameters and a dummy queue for FeedbackUI
    selected_vein = "Left Vein"
    selected_point = "Point A"
    work_queue = Queue()  # Provide an empty queue if needed

    feedback_ui = FeedbackUI(selected_vein, selected_point, work_queue=sig_processed)
    feedback_ui.show()

    """"
    # For demonstration, here is some dummy viewer perspective data:
    # This data might be received from an external source.
    viewer_rvec = np.array([[-0.0681944 ,  2.67529694,  0.22829397]])
    viewer_tvec = np.array([[162.8271214 ,  70.32294188, 169.37338088]])"
    """
    
    
    # Set up a QTimer to update the projection capture periodically (every 100 ms)
    # When viewer_rvec and viewer_tvec are provided, the image center will align with the viewer's line of sight.
    timer = QTimer()
    timer.timeout.connect(lambda: update_display(feedback_ui, angle_deg=25, viewer_rvec=viewer_rvec, viewer_tvec=viewer_tvec))
    timer.start(100)  # Update every 100 milliseconds

    sys.exit(app.exec())

if __name__ == "__main__":
    start(sig_processed=Queue())
