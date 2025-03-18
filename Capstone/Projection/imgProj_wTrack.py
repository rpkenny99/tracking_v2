import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
from queue import Queue
import os
sys.path.append(os.path.join("Capstone", "Tracking"))
import markerDetectionFrame
import queue

# Import the UI from feedback3.py (adjust the import path as needed)
from feedback3 import FeedbackUI

def qpixmap_to_numpy(pixmap):
    """
    Convert a QPixmap to a NumPy array (BGR format for OpenCV).
    """
    # Convert QPixmap to QImage using the proper enum for RGBA8888.
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)
    # Convert from RGBA to BGR (which OpenCV expects)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

def adjust_for_head_pose(image, rvec, tvec, fixed_obj, f, scale=0.922):
    """
    Adjust the image based on the user's head pose so that the projected position
    of the fixed physical object is centered in the display.

    Parameters:
      image: The input image (numpy array) from the UI.
      rvec: The head orientation as a Rodrigues vector (3,).
      tvec: The head position (translation vector, 3,).
      fixed_obj: The fixed physical object coordinates (3,) in the same world space.
      f: Focal length in pixel units.
      scale: Scaling factor (here ~0.922 corresponds to a 15% reduction in area).
    """
    # Convert rvec to a rotation matrix.
    R, _ = cv2.Rodrigues(rvec)
    # Invert the rotation to get the transformation from world coordinates to head coordinates.
    R_inv = R.T
    # Compute the fixed object's position in the head coordinate system.
    rel_obj = R_inv @ (fixed_obj.reshape(3, 1) - tvec.reshape(3, 1))
    rel_obj = rel_obj.flatten()  # Now shape (3,)

    # Perspective projection (assuming a pinhole camera model)
    proj_x = f * (rel_obj[0] / rel_obj[2])
    proj_y = f * (rel_obj[1] / rel_obj[2])

    # Compute the center of the image.
    h, w = image.shape[:2]
    center_x, center_y = w / 2, h / 2

    # To center the projected object, we need to translate the image by:
    tx = center_x - proj_x
    ty = center_y - proj_y

    # Construct the affine transformation matrix that applies scaling and translation.
    M = np.array([
        [scale, 0, tx],
        [0, scale, ty]
    ], dtype=np.float32)

    adjusted = cv2.warpAffine(image, M, (w, h))
    return adjusted

def main():
    app = QApplication(sys.argv)

    # Create an empty work queue to pass into FeedbackUI.
    my_queue = Queue()
    selected_vein = "Left Vein"
    selected_point = "Point A"

    # Instantiate FeedbackUI from feedback3.py (adjust if additional parameters are needed)
    feedback_ui = FeedbackUI(selected_vein, selected_point, work_queue=my_queue)
    feedback_ui.show()

    # Process events to ensure the window is rendered.
    app.processEvents()

    # Capture the UI as a QPixmap and convert it to a NumPy array.
    pixmap = feedback_ui.grab()
    original_image = qpixmap_to_numpy(pixmap)

    # --- Example head pose data ---
    # These values come from your tracking system.
    rvec = np.array([0.18123738, 3.02991718, 0.14528933], dtype=np.float32)
    tvec = np.array([116.98430701, 84.41693627, 186.72209475], dtype=np.float32)

    # Fixed physical object position in the same coordinate space (example values).
    fixed_obj = np.array([100, 80, 200], dtype=np.float32)

    # Focal length in pixel units (this value must be calibrated for your system).
    focal_length = 800.0  # Example value

    # Adjust the image: scale down by ~15% area and translate so the physical object aligns.
    adjusted_image = adjust_for_head_pose(original_image, rvec, tvec, fixed_obj, focal_length)

    # Display the adjusted image (this simulates what the projector would show).
    cv2.imshow("Adjusted Projection", adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
