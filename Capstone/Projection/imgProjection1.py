import cv2
import numpy as np
import math
import updated_feeback  # Hypothetical module providing real-time frames
# Capstone/Projection/updated_feeback.py
# Global variable to store the inverse matrix

# other imports
import sys
# import os
# sys.path.append(os.path.join("Capstone", "SignalProcessing"))
# import signal_processing



inverse_transform_matrix = None

def map_point(point, inv_matrix):
    """
    Map a point from warped image space back to original space using the inverse transform.
    """
    x, y = point
    pt = np.array([x, y, 1.0]).reshape((3, 1))
    orig_pt = np.dot(inv_matrix, pt)
    orig_pt /= orig_pt[2]  # Normalize
    return int(orig_pt[0]), int(orig_pt[1])

def mouse_callback(event, x, y, flags, param):
    """
    Mouse event callback that maps the mouse coordinates from the warped display back
    to the original image coordinates.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map the clicked point back to original coordinates.
        orig_x, orig_y = map_point((x, y), inverse_transform_matrix)
        print(f"Clicked on warped image at ({x}, {y}) -> original coords: ({orig_x}, {orig_y})")
        # Here, you could check if the click is within a UI element region in the original image.
        # For example:
        # if ui_button_rect.contains((orig_x, orig_y)):
        #     perform_action()
        
def main():
    global inverse_transform_matrix

    # Get an initial frame to determine dimensions
    test_frame = updated_feeback.get_current_frame()
    if test_frame is None:
        print("No frame received from updated_feedback.")
        return

    h, w = test_frame.shape[:2]

    # Define source points (corners of the original image)
    src_points = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    # Define destination points to simulate a 25° tilt
    angle_deg = 25.0
    angle_rad = math.radians(angle_deg)
    offset = h * math.tan(angle_rad)

    dst_points = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1 + offset, h - 1],
        [0 + offset, h - 1]
    ])

    # Compute perspective transform and its inverse
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    # Set up OpenCV window and mouse callback for interactive UI
    window_name = "Interactive Augmented Overlay"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        frame = updated_feeback.get_current_frame()
        if frame is None:
            print("No frame. Exiting.")
            break

        # Apply the perspective warp
        warped_frame = cv2.warpPerspective(frame, transform_matrix, (w, h))

        # Display the warped image
        cv2.imshow(window_name, warped_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
