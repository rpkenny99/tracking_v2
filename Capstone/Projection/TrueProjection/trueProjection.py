import sys
import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
# from feedback3 import FeedbackUI  # Import your UI class
from queue import Queue


def rodrigues_to_mat(rvec, tvec):
    """
    Convert (rvec, tvec) into a 4x4 homogeneous transform.
    - rvec: (3,) rotation vector (Rodrigues)
    - tvec: (3,) translation vector
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3]  = tvec
    return T

def project_points(points_3d, view_matrix, camera_matrix):
    """
    Project 3D points into 2D using a pinhole camera model:
      X_cam = view_matrix * X_world
      uv    = camera_matrix * X_cam
    Returns Nx2 pixel coordinates.
    """
    # Convert to homogeneous
    num_pts = points_3d.shape[0]
    homog_pts = np.hstack([points_3d, np.ones((num_pts, 1), dtype=np.float32)])
    
    # Transform into camera space
    pts_cam = (view_matrix @ homog_pts.T).T
    
    # Apply intrinsics
    uv = (camera_matrix @ pts_cam[:, :3].T).T
    
    # Normalize
    uv_2d = np.zeros((num_pts, 2), dtype=np.float32)
    uv_2d[:, 0] = uv[:, 0] / (uv[:, 2] + 1e-9)
    uv_2d[:, 1] = uv[:, 1] / (uv[:, 2] + 1e-9)
    
    return uv_2d

def compute_homography(src_pts, dst_pts):
    """
    Compute a 3x3 homography mapping src_pts -> dst_pts.
    Both arrays must be Nx2 in float32.
    """
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H


"""# ------------------------------------------------------------------------------
# Main function that updates the display for the glass
# ------------------------------------------------------------------------------"""
def update_display_for_glass(
    monitor_img,
    user_rvec, user_tvec,           # changes in real-time
    glass_rvec, glass_tvec,         # typically static
    object_rvec, object_tvec,       # typically static
    object_size,                    # (width, height)
    camera_matrix
):
    """
    Returns a warped image that appears aligned with the physical object 
    behind the tinted glass, from the user's perspective.
    """
    # 1. Build 4x4 transforms
    T_user   = rodrigues_to_mat(user_rvec,   user_tvec)
    T_glass  = rodrigues_to_mat(glass_rvec,  glass_tvec)   # might be used if needed
    T_object = rodrigues_to_mat(object_rvec, object_tvec)

    # 2. Define the corners of the 2D monitor image in pixel coords
    H_img, W_img = monitor_img.shape[:2]
    src_corners = np.array([
        [0,      0     ],
        [W_img,  0     ],
        [W_img,  H_img ],
        [0,      H_img ]
    ], dtype=np.float32)

    # 3. Define the 3D corners of the object's "front face" in local coords
    #    For example, a rectangle of size (width=object_size[0], height=object_size[1]) at z=0
    obj_w, obj_h = object_size
    object_local_corners = np.array([
        [0,      0,      0],
        [obj_w,  0,      0],
        [obj_w,  obj_h,  0],
        [0,      obj_h,  0]
    ], dtype=np.float32)

    # 4. Transform these corners to world space
    num_pts = object_local_corners.shape[0]
    homog_obj = np.hstack([object_local_corners, np.ones((num_pts, 1), dtype=np.float32)])
    obj_world = (T_object @ homog_obj.T).T  # Nx4

    # 5. Project into user's view
    T_user_inv = np.linalg.inv(T_user)   # user camera = inverse of (user->world)
    dst_corners = project_points(obj_world[:, :3], T_user_inv, camera_matrix)

    # 6. Compute homography from monitor corners -> object's corners in user view
    H_mat = compute_homography(src_corners, dst_corners)

    # 7. Warp the monitor image to align with the object
    #    For simplicity, choose an output size that fits the user’s camera FOV
    out_width  = int(camera_matrix[0,2] * 2)  # e.g. 2 * cx
    out_height = int(camera_matrix[1,2] * 2)  # e.g. 2 * cy
    warped = cv2.warpPerspective(monitor_img, H_mat, (out_width, out_height))

    return warped





"""Home Function"""
def main_loop():
    # Camera intrinsics for user viewpoint (example)
    camera_matrix = np.array([
        [800,   0, 640],
        [  0, 800, 360],
        [  0,   0,   1]
    ], dtype=np.float32)
    
    # Known (static) data
    glass_rvec  = np.array([ 0.0,  0.0,  0.0], dtype=np.float32)  # example
    glass_tvec  = np.array([10.0,  0.0,  0.0], dtype=np.float32)  # example
    object_rvec = np.array([ 0.0,  0.0,  0.0], dtype=np.float32)
    object_tvec = np.array([ 0.0,  0.0, 50.0], dtype=np.float32)  # 50 mm away, for instance
    object_size = (100.0, 100.0)  # 100 mm x 100 mm
    
    # Suppose we have a function get_monitor_image() that grabs the current monitor UI
    # We'll just read a placeholder image for this demo
    monitor_img = cv2.imread("some_test_image.png")  # placeholder

    while True:
        # 1. Get the current user pose (rvec, tvec) from your tracking system
        user_rvec, user_tvec = get_user_pose()  # e.g., from your thread or global variable

        # 2. Warp the monitor image to appear on the glass
        warped_img = update_display_for_glass(
            monitor_img,
            user_rvec, user_tvec,
            glass_rvec, glass_tvec,
            object_rvec, object_tvec,
            object_size,
            camera_matrix
        )

        # 3. Display the warped image (this is what you'd show in a window)
        cv2.imshow("AR Overlay", warped_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()
