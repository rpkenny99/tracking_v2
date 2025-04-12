import cv2
import numpy as np
import time

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def rodrigues_to_mat(rvec, tvec):
    """
    Convert (rvec, tvec) from an optical tracker or similar
    into a 4x4 homogeneous transformation matrix.
    rvec: (3,) rotation vector (Rodrigues)
    tvec: (3,) translation vector
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3]  = np.array(tvec, dtype=np.float32).reshape(3)
    return T

def project_points(points_3d, view_matrix, camera_matrix):
    """
    Given a set of 3D points, a 4x4 view matrix (world->camera),
    and an intrinsic camera matrix (3x3), return 2D projections.
    This is typical pinhole projection:
       X_cam = view_matrix * X_world
       x_pix = camera_matrix * X_cam
    We'll ignore lens distortion for simplicity.
    """
    # Convert points to homogeneous coords
    num_pts = points_3d.shape[0]
    homog_pts = np.hstack([points_3d, np.ones((num_pts,1), dtype=np.float32)])
    
    # Transform points into camera frame
    pts_cam = (view_matrix @ homog_pts.T).T
    
    # Project using camera intrinsics
    uv = (camera_matrix @ pts_cam[:,:3].T).T
    
    # Convert to normalized 2D
    # uv[:,0] = X * fx / Z + cx, etc.
    uv_2d = np.zeros((num_pts,2), dtype=np.float32)
    uv_2d[:,0] = uv[:,0] / (uv[:,2] + 1e-9)
    uv_2d[:,1] = uv[:,1] / (uv[:,2] + 1e-9)
    
    return uv_2d

def compute_homography(src_pts, dst_pts):
    """
    Compute the 3x3 homography matrix H that maps
    src_pts[i] -> dst_pts[i], i.e.  H * src = dst
    Both src_pts and dst_pts should be N×2 arrays.
    """
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H

# ------------------------------------------------------------------------------
# Main function that updates the display for the glass
# ------------------------------------------------------------------------------

def update_display_for_glass(monitor_img,
                             user_rvec, user_tvec,
                             glass_rvec, glass_tvec,
                             monitor_rvec, monitor_tvec,
                             object_rvec, object_tvec,
                             object_size,
                             camera_matrix):
    """
    - monitor_img:  The image we want to warp (from the monitor’s UI, for instance).
    - user_rvec, user_tvec: Pose of the user's eye or head in world coordinates.
    - glass_rvec, glass_tvec: Pose of the tinted glass plane in world coordinates.
    - monitor_rvec, monitor_tvec: Pose of the monitor in world coordinates (if needed).
    - object_rvec, object_tvec: Pose of the physical object in world coordinates.
    - object_size:  Known physical dimensions (e.g., width, height) of the object.
    - camera_matrix: Intrinsic parameters for the “virtual camera” from the user’s perspective.
    
    Returns a warped image that you can display with cv2.imshow.
    """
    
    # 1. Build transformation matrices for each
    T_user    = rodrigues_to_mat(user_rvec, user_tvec)
    T_glass   = rodrigues_to_mat(glass_rvec, glass_tvec)
    T_monitor = rodrigues_to_mat(monitor_rvec, monitor_tvec)
    T_object  = rodrigues_to_mat(object_rvec, object_tvec)
    
    # 2. Define the corners of the "monitor image" in some local 2D coords.
    #    For example, if monitor_img is W x H:
    H_img, W_img = monitor_img.shape[:2]
    src_corners = np.array([
        [0,     0    ],
        [W_img, 0    ],
        [W_img, H_img],
        [0,     H_img]
    ], dtype=np.float32)
    
    # 3. Figure out where these corners should map onto the glass plane so that
    #    from the user’s perspective, they align with the real object behind it.
    #
    #    Simplest approach (conceptually):
    #    - Identify the 3D corners of the object in world space
    #      (for instance, the "front face" of the object if you want to overlay).
    #    - Project those corners into the user's camera to see where they appear in 2D.
    #    - That 2D region is your "destination" for the warped image.
    #
    #    Below is a toy example that just uses object_size to define corners in local object coords:
    
    obj_w, obj_h = object_size  # e.g., (width, height)
    # Let's say the object's local corners are at z=0 in object space:
    object_local_corners = np.array([
        [0,      0,      0],
        [obj_w,  0,      0],
        [obj_w,  obj_h,  0],
        [0,      obj_h,  0]
    ], dtype=np.float32)
    
    # 4. Convert object local corners to world space
    num_pts = object_local_corners.shape[0]
    homog_obj = np.hstack([object_local_corners, np.ones((num_pts,1), dtype=np.float32)])
    T_obj_world = T_object  # 4x4
    obj_world = (T_obj_world @ homog_obj.T).T  # Nx4
    
    # 5. Now project these points into the user’s camera frame
    #    We need the user’s "view matrix" from world -> user_camera.
    #    Typically, if T_user is user->world, we want the inverse for world->user.
    T_user_inv = np.linalg.inv(T_user)
    
    # We'll also need a 3x3 camera_matrix for the user’s viewpoint.
    # The function below returns Nx2 pixel coordinates for Nx3 input points.
    dst_corners = project_points(obj_world[:, :3], T_user_inv, camera_matrix)
    # Now we have a 2D polygon in user’s “view space” where we want to map our monitor image.
    
    # 6. Compute the homography that warps the monitor image corners to the object corners
    H_mat = compute_homography(src_corners, dst_corners)
    
    # 7. Warp the monitor image so that it appears at the correct place from the user’s perspective
    #    We'll pick an output size that fits the user’s entire view or the bounding box of dst_corners.
    #    For simplicity, let's just use the same size as the original monitor image or bigger.
    out_size = (int(camera_matrix[0,2]*2), int(camera_matrix[1,2]*2))  
    # i.e., a guess that the user’s camera view might be ~2*cx by 2*cy
    
    warped = cv2.warpPerspective(monitor_img, H_mat, out_size)
    
    return warped


# ------------------------------------------------------------------------------
# Example usage in a loop
# ------------------------------------------------------------------------------

def main_loop():
    # Suppose you have a function get_poses() that returns the current
    # rvec, tvec for user, glass, monitor, object, etc. in real time.
    # Also a function get_monitor_image() that grabs the monitor’s content as an image.
    # And a known camera_matrix for the user’s perspective.
    
    camera_matrix = np.array([
        [800,   0, 640],  # fx,  0, cx
        [  0, 800, 360],  # 0,  fy, cy
        [  0,   0,   1]
    ], dtype=np.float32)
    
    object_size = (200, 100)  # 200 mm wide, 100 mm tall, for example
    
    while True:
        # 1. Acquire the latest poses
        user_rvec, user_tvec, \
        glass_rvec, glass_tvec, \
        monitor_rvec, monitor_tvec, \
        object_rvec, object_tvec = get_poses()  # user-defined function

        # 2. Get the latest monitor image
        monitor_img = get_monitor_image()  # e.g. a screenshot or UI capture

        # 3. Warp the display for the glass
        warped_img = update_display_for_glass(monitor_img,
                                              user_rvec, user_tvec,
                                              glass_rvec, glass_tvec,
                                              monitor_rvec, monitor_tvec,
                                              object_rvec, object_tvec,
                                              object_size,
                                              camera_matrix)

        # 4. Show or render the result
        cv2.imshow("Augmented Overlay", warped_img)
        
        # Press ESC or Q to quit
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break
        
        time.sleep(0.01)  # small delay to avoid busy loop

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
