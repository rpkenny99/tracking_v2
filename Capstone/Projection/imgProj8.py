import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QScreen
from PyQt6.QtGui import QImage
from feedback3 import FeedbackUI

def qpixmap_to_numpy(pixmap):
    """
    Convert a QPixmap to a NumPy array (in BGR format for OpenCV).
    """
    # Convert QPixmap to QImage and then ensure it is in a known format (RGBA8888)
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)  # QImage.Format_RGBA8888
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)
    # Convert from RGBA to BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

def apply_projection_transform(image, angle_deg=25):
    """
    Applies a perspective transformation to the given image to simulate
    the view of a projection plane rotated by angle_deg relative to the screen.
    
    For this example, we assume that the rotation is around the horizontal axis,
    which causes the top of the image to be foreshortened.
    """
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle_deg)

    # Calculate the new width of the top edge (foreshortened by cosine of the angle)
    new_top_width = w * np.cos(angle_rad)
    # Calculate horizontal offset so that the top edge is centered
    offset = (w - new_top_width) / 2

    # Define source points (corners of the original image)
    src_pts = np.float32([[0, 0],
                          [w, 0],
                          [w, h],
                          [0, h]])
    # Define destination points to simulate the projection.
    # The top edge is "squeezed" inward by the offset.
    dst_pts = np.float32([[offset, 0],
                          [w - offset, 0],
                          [w, h],
                          [0, h]])

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # Apply the transformation
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def main():
    app = QApplication(sys.argv)

    # Grab the entire primary screen (or if you want just your UI, grab its window ID)
    screen: QScreen = app.primaryScreen()
    pixmap = screen.grabWindow(0)  # Replace 0 with your window ID if needed
    original_image = qpixmap_to_numpy(pixmap)

    # Apply the projection transformation for a 25° tilt
    transformed_image = apply_projection_transform(original_image, angle_deg=25)

    # Display the transformed image (this window simulates what the projector shows)
    cv2.imshow("Projected UI", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
