import cv2
import numpy as np

def rvec_to_mat(rvec):
    """
    Convert a rotation vector (rvec) to a rotation matrix.
    """
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float32))
    return R

def project_point_to_plane(user_tvec, target_point, plane_point, plane_normal):
    """
    Given a ray starting at 'user_tvec' and going in the direction of (target_point - user_tvec),
    compute its intersection with a plane defined by plane_point and plane_normal.
    """
    direction = target_point - user_tvec
    t = np.dot(plane_point - user_tvec, plane_normal) / np.dot(direction, plane_normal)
    intersection = user_tvec + t * direction
    return intersection

def world_to_glass_coords(world_point, glass_tvec, glass_R):
    """
    Convert a point in world coordinates to 2D coordinates on the glass plane.
    Here we assume the glass coordinate system is defined by glass_R and glass_tvec,
    with the glass lying on the z=0 plane in its local coordinate system.
    """
    local_point = np.dot(glass_R.T, (world_point - glass_tvec))
    # Return x, y as coordinates on the glass
    return local_point[:2]

def compute_homography(user_tvec, glass_tvec, glass_rvec, monitor_tvec, monitor_rvec, monitor_image):
    """
    Given the user’s dynamic position (user_tvec), the static glass and monitor poses,
    and the monitor image, compute the homography that warps the monitor image so that its
    projection onto the glass (as seen by the user) aligns with the physical object.
    """
    # Convert rotation vectors to rotation matrices.
    glass_R = rvec_to_mat(glass_rvec)
    monitor_R = rvec_to_mat(monitor_rvec)
    
    # Get monitor image dimensions.
    h, w = monitor_image.shape[:2]
    # Define monitor image corners in 2D (monitor coordinate system)
    src_pts = np.array([[0, 0],
                        [w, 0],
                        [w, h],
                        [0, h]], dtype=np.float32)
    
    # Convert these 2D monitor corners to 3D world coordinates.
    # (Assuming the monitor’s image plane lies at z=0 in its local coordinate frame.)
    world_pts = []
    for pt in src_pts:
        pt_3d = np.array([pt[0], pt[1], 0], dtype=np.float32)
        world_point = np.dot(monitor_R, pt_3d) + monitor_tvec
        world_pts.append(world_point)
    world_pts = np.array(world_pts, dtype=np.float32)

    # Define the glass plane.
    # Assume that in the glass coordinate system the plane is z=0;
    # thus, in world coordinates the plane normal is:
    glass_normal = np.dot(glass_R, np.array([0, 0, 1], dtype=np.float32))
    
    # For each world point from the monitor, project along the ray from the user onto the glass.
    dest_pts = []
    for world_pt in world_pts:
        # Compute intersection on the glass plane.
        intersection = project_point_to_plane(user_tvec, world_pt, glass_tvec, glass_normal)
        # Convert the intersection to 2D coordinates in the glass's coordinate system.
        local_pt = world_to_glass_coords(intersection, glass_tvec, glass_R)
        dest_pts.append(local_pt)
    dest_pts = np.array(dest_pts, dtype=np.float32)
    
    # Compute the homography mapping the monitor image corners (src_pts)
    # to the destination points (dest_pts) on the glass.
    H, status = cv2.findHomography(src_pts, dest_pts)
    return H, dest_pts

def main():
    # Load (or create) the monitor image.
    monitor_image = cv2.imread("monitor_display.jpg")
    if monitor_image is None:
        # Create a dummy image if none is found.
        monitor_image = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.putText(monitor_image, "Monitor Display", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # --------------------------
    # Set up static parameters.
    # --------------------------
    # Glass parameters: we assume the glass is at the origin and rotated 25° about the x-axis.
    glass_tvec = np.array([0, 0, 0], dtype=np.float32)      
    glass_rvec = [np.deg2rad(25), 0, 0]  # 25 degrees about x-axis.
    
    # Monitor parameters: position and orientation in world space.
    monitor_tvec = np.array([0, -0.5, -0.5], dtype=np.float32)   
    monitor_rvec = [0, 0, 0]  # Monitor is parallel to the glass.
    
    # Physical object parameters (for reference, though not used directly in the warping here):
    physical_object_tvec = np.array([0, 0, 2], dtype=np.float32)  # Example: 2 meters away.
    physical_object_size = 0.5   # For instance, half a meter; used when matching size.
    
    # --------------------------
    # Set up dynamic user pose.
    # --------------------------
    # Initially, we place the user 1 meter in front of the glass (assuming glass is at z=0).
    user_tvec = np.array([0, 0, -1], dtype=np.float32)
    user_rvec = [0, 0, 0]  # Assume the user is initially looking straight ahead.
    
    # -------------
    # Main loop.
    # -------------
    # (In a real system, these values would come from your tracking system.)
    while True:
        # Simulate an update of the user's position in real time.
        # For example, we oscillate the x-position slightly.
        user_tvec[0] = 0.1 * np.sin(cv2.getTickCount() / cv2.getTickFrequency())
        
        # Compute the homography matrix based on the current user pose.
        H, dest_pts = compute_homography(user_tvec, glass_tvec, glass_rvec,
                                         monitor_tvec, monitor_rvec, monitor_image)
        
        # To display the warped monitor image on the glass,
        # we define a destination image size based on the destination points.
        min_pt = np.min(dest_pts, axis=0)
        max_pt = np.max(dest_pts, axis=0)
        width = int(np.ceil(max_pt[0] - min_pt[0]))
        height = int(np.ceil(max_pt[1] - min_pt[1]))
        if width <= 0 or height <= 0:
            width, height = monitor_image.shape[1], monitor_image.shape[0]
        
        warped = cv2.warpPerspective(monitor_image, H, (width, height))
        
        cv2.imshow("Transformed Display on Glass", warped)
        key = cv2.waitKey(30)  # Update roughly every 30ms.
        if key == 27:  # Press ESC to exit.
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
