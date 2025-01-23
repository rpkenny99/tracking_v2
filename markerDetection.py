from asyncio import sleep
import cv2 as cv2
import numpy as np
from cv2 import aruco
import os
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
import geometric_rotations as gr
import json

markerDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
paramMarkers = aruco.DetectorParameters()
calib_data_path = "calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

last_4_poses = []
prev_pose = []

last_4_positions = []
prev_position = []

consequtive_failures = 0

dodecahedron_edge_length_mm = 14
inradius_mm = math.sqrt((25 + 11*math.sqrt(5))/40) * dodecahedron_edge_length_mm

MARKER_SIZE = 12
REFERENCE_RVEC = np.array([1.2639763 ,  1.45933559, -1.29766979])
REFERENCE_TVEC = np.array([95.22124999,  73.81633065, 304.82649138])
FPS = 30
TIME_PER_FRAME = 1/FPS

# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

plt.ion()
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

# plt.ion()
# fig3 = plt.figure()
# ax3 = fig2.add_subplot(111, projection='3d')

def DisplayFrame(frame):
    cv2.imshow("preview", frame)

def calculate_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.squeeze()

    error = np.linalg.norm(image_points - projected_points, axis=1)
    mean_error = np.mean(error)

    return mean_error

def transform_corners_to_world(local_corners, rVec_world, tVec_world):
    """
    Transforms the marker corners from the marker's local coordinate system to the world coordinate system.

    :param local_corners: 3D coordinates of the marker corners in the marker's local coordinate system (4x3 array).
    :param rVec_world: Rotation vector representing the orientation of the marker in the world coordinate system (3x1).
    :param tVec_world: Translation vector representing the position of the marker in the world coordinate system (3x1).
    :return: Array of 3D world coordinates for each corner.
    """
    # Convert the rotation vector to a rotation matrix
    R_world, _ = cv2.Rodrigues(rVec_world)

    # Transform each local corner to the world coordinate system
    world_corners = []
    for corner in local_corners:
        # Apply rotation and translation to get world coordinates
        world_point = R_world @ corner.reshape(3, 1) + tVec_world.reshape(3, 1)
        world_corners.append(world_point.flatten())

    return np.array(world_corners)


def transform_to_world(rVec, tVec, rVec_origin, tVec_origin):
    """
    Transforms the pose of a marker from camera coordinates to world coordinates.
    The world coordinate system is defined by the reference marker (rVec_origin, tVec_origin).

    :param rVec: Rotation vector of the marker (3x1).
    :param tVec: Translation vector of the marker (3x1).
    :param rVec_origin: Rotation vector of the reference marker (3x1).
    :param tVec_origin: Translation vector of the reference marker (3x1).
    :return: Transformed rotation vector and translation vector in world coordinates.
    """
    # Convert the reference marker's rotation vector to a rotation matrix
    R_origin, _ = cv2.Rodrigues(rVec_origin)
    t_origin = tVec_origin.reshape(3, 1)

    # Invert the reference marker's transformation to get camera-to-world transformation
    R_origin_inv = R_origin.T
    t_origin_inv = -R_origin_inv @ t_origin

    # Convert the marker's rotation vector to a rotation matrix
    R_marker, _ = cv2.Rodrigues(rVec)
    t_marker = tVec.reshape(3, 1)

    # Transform the marker's pose from camera coordinates to world coordinates
    # R_world = R_origin_inv * R_marker
    # t_world = R_origin_inv * t_marker + t_origin_inv
    R_world = R_origin_inv @ R_marker
    t_world = R_origin_inv @ t_marker + t_origin_inv

    # Convert the world rotation matrix back to a rotation vector
    rVec_world, _ = cv2.Rodrigues(R_world)

    return rVec_world, t_world

def plot_marker_corners(corners):
    """
    Plots the 3D coordinates of marker corners. Each time this function is called,
    the plot is updated with the new data.

    :param corners: A list of sets of corners, where each set contains the 3D coordinates of the corners.
                    The input should be in the form [[(x1, y1, z1), (x2, y2, z2), ...], ...] for multiple markers.
    """
    # Clear the current plot
    ax.cla()

    # Loop through each marker's set of corners
    for set_of_corners in corners:
        # Extract x, y, z coordinates for the current set of corners
        x = [corner[0] for corner in set_of_corners]
        y = [corner[1] for corner in set_of_corners]
        z = [corner[2] for corner in set_of_corners]

        # Plot the corners for this marker
        ax.scatter(x, y, z, marker='o', s=50, label='Marker Corners')

        # Optionally, you can connect the corners to form the marker's shape
        # Closing the loop to form a quadrilateral
        ax.plot(x + [x[0]], y + [y[0]], z + [z[0]], 'b-')

    # Set axis labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D Plot of Marker Corners')

    # Display legend
    ax.legend()

    # Draw the updated plot and flush events
    plt.draw()
    fig.canvas.flush_events()

def resolve_quaternion_signs(quaternions):
    # Ensure all quaternions are in the same hemisphere
    reference = quaternions[0]
    for i in range(len(quaternions)):
        if np.dot(reference, quaternions[i]) < 0:
            quaternions[i] = -quaternions[i]
    return quaternions

def compute_average_pose(rVec_total):
    rotations = [R.from_matrix(pose[:3, :3]) for pose in rVec_total]
    
    # Average rotations using quaternions
    quaternions = np.array([rot.as_quat() for rot in rotations])
    quaternions = resolve_quaternion_signs(quaternions)
    avg_quaternion = np.mean(quaternions, axis=0)
    avg_quaternion /= np.linalg.norm(avg_quaternion)  # Normalize
    avg_rotation = R.from_quat(avg_quaternion)
    
    return avg_rotation.as_matrix()

def compute_angular_deviations(avg_pose, poses):
    avg_rotation = R.from_matrix(avg_pose[:3, :3])
    deviations = []
    for pose in poses:
        rotation = R.from_matrix(pose[:3, :3])
        relative_rotation = avg_rotation.inv() * rotation
        angular_distance = relative_rotation.magnitude()
        deviations.append(angular_distance)
    return np.array(deviations)

def detect_faulty_poses_fixed_threshold(poses, deviations, markerIds, threshold=0.7):
    # Collect filtered (valid) and faulty poses based on the fixed threshold
    filtered_poses=[]
    faulty_poses=[]
    faulty_marker_ids=[]
    
    for pose, dev, id in zip(poses, deviations, markerIds):
        if dev <= threshold:
            filtered_poses.append(pose)
        else:
            faulty_poses.append(pose)
            faulty_marker_ids.append(id)
    return filtered_poses, faulty_poses, faulty_marker_ids

def detect_faulty_poses(poses, deviations, markerIds, n_std=1.25):
    mean_dev = np.mean(deviations)
    std_dev = np.std(deviations)
    threshold = mean_dev + n_std * std_dev

    filtered_poses = []
    faulty_poses = []
    faulty_marker_ids = []
    
    # Filter out poses with deviations above the threshold
    for pose, dev, id in zip(poses, deviations, markerIds):
        if dev <= threshold:
            filtered_poses.append(pose)
        else:
            faulty_poses.append(pose)
            faulty_marker_ids.append(id)
    return filtered_poses, faulty_poses, faulty_marker_ids

def plot_marker_rotation(avg_pose, rotated_poses):
    global ax2
    ax2.cla()  # Clear the current plot to reuse the same figure and axis

    scale_factor = 0.05

    if rotated_poses is not None:
        for rotMat in rotated_poses:
            # Define the origin and scaled axes as column vectors
            origin = np.array([0, 0, 0])
            x_axis = scale_factor * np.array([1, 0, 0]).reshape(3, 1)
            y_axis = scale_factor * np.array([0, 1, 0]).reshape(3, 1)
            z_axis = scale_factor * np.array([0, 0, 1]).reshape(3, 1)

            # Apply the rotation to each scaled axis and flatten the result to get 1D arrays
            x_rotated = (rotMat @ x_axis).flatten()
            y_rotated = (rotMat @ y_axis).flatten()
            z_rotated = (rotMat @ z_axis).flatten()

            # Plot the rotated axes with short lines
            ax2.quiver(*origin, *x_rotated, color='r', linestyle='dashed', label='Rotated X')
            ax2.quiver(*origin, *y_rotated, color='g', linestyle='dashed', label='Rotated Y')
            ax2.quiver(*origin, *z_rotated, color='b', linestyle='dashed', label='Rotated Z')

    # saved_markers = fix_faulty_markers(faulty_markers, faulty_marker_ids)
    # saved_markers = adjust

    if avg_pose is not None:
        # Define the origin and scaled axes as column vectors
        origin = np.array([0, 0, 0])
        x_axis = scale_factor * np.array([1, 0, 0]).reshape(3, 1)
        y_axis = scale_factor * np.array([0, 1, 0]).reshape(3, 1)
        z_axis = scale_factor * np.array([0, 0, 1]).reshape(3, 1)

        # Apply the rotation to each scaled axis and flatten the result to get 1D arrays
        x_rotated = (avg_pose @ x_axis).flatten()
        y_rotated = (avg_pose @ y_axis).flatten()
        z_rotated = (avg_pose @ z_axis).flatten()

        # Plot the rotated axes with short lines
        ax2.quiver(*origin, *x_rotated, color='r', linestyle='solid', label='Rotated X')
        ax2.quiver(*origin, *y_rotated, color='g', linestyle='solid', label='Rotated Y')
        ax2.quiver(*origin, *z_rotated, color='b', linestyle='solid', label='Rotated Z')

    # Only add the legend once
    ax2.legend()
    plt.draw()
    fig2.canvas.flush_events()

def plot_singular_points(points):
    """
    Plots singular 3D points. Each time this function is called,
    the plot is updated with the new data.

    :param points: A list of 3D points in the form [(x1, y1, z1), (x2, y2, z2), ...]
    """
    # Clear the current plot
    global ax

    ax.cla()

    # Extract x, y, z coordinates from the list of points
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    # Plot the singular points
    ax.scatter(x, y, z, marker='o', s=50, c='r', label='Singular Points')

    # Draw the updated plot and flush events
    plt.draw()
    fig.canvas.flush_events()

def compute_velocity(rvec_prev, tvec_prev, rvec_curr, tvec_curr, delta_t):
    # Compute translational velocity
    v_x = (tvec_curr[0] - tvec_prev[0]) / delta_t
    v_y = (tvec_curr[1] - tvec_prev[1]) / delta_t
    v_z = (tvec_curr[2] - tvec_prev[2]) / delta_t

    # Convert rvec to rotation matrices
    R_curr, _ = cv2.Rodrigues(rvec_curr)
    R_prev, _ = cv2.Rodrigues(rvec_prev)

    # Compute relative rotation matrix
    R_relative = np.dot(R_curr, R_prev.T)

    # Convert relative rotation matrix back to rotation vector
    rvec_relative, _ = cv2.Rodrigues(R_relative)

    # Compute angular velocity
    omega_x = rvec_relative[0][0] / delta_t
    omega_y = rvec_relative[1][0] / delta_t
    omega_z = rvec_relative[2][0] / delta_t

    return (v_x, v_y, v_z), (omega_x, omega_y, omega_z)

def get_average_pose(rVec_total, markerIds):
    global last_4_poses
    global consequtive_failures

    # Define a scaling factor for short lines (adjust as needed)
    scale_factor = 0.05
    rotated_poses = []
    for rVec, id in zip(rVec_total, markerIds):
        rotMat, _ = cv2.Rodrigues(rVec)
        if id[0] == 0:
            pass
        elif id[0] == 1:
            rotMat = gr.rotate_marker_1(rotMat)
        elif id[0] == 2:
            rotMat = gr.rotate_marker_2(rotMat)
        elif id[0] == 3:
            rotMat = gr.rotate_marker_3(rotMat)
        elif id[0] == 4:
            rotMat = gr.rotate_marker_4(rotMat)
        elif id[0] == 5:
            rotMat = gr.rotate_marker_5(rotMat)
        elif id[0] == 6:
            rotMat = gr.rotate_marker_6(rotMat)
        elif id[0] == 7:
            rotMat = gr.rotate_marker_7(rotMat)
        elif id[0] == 8:
            rotMat = gr.rotate_marker_8(rotMat)
        elif id[0] == 9:
            rotMat = gr.rotate_marker_9(rotMat)
        elif id[0] == 14:
            rotMat = gr.rotate_marker_14(rotMat)
        elif id[0] == 19:
            rotMat = gr.rotate_marker_19(rotMat)
        else:
            continue

        
        if len(last_4_poses) == 4:
            previous_pose_avg = compute_average_pose(last_4_poses)
            deviations = compute_angular_deviations(previous_pose_avg, [rotMat])
            filtered_poses, _, _ = detect_faulty_poses_fixed_threshold([rotMat], deviations, markerIds)
            if len(filtered_poses) != 0:
                print(f"Marker ID {id[0]} passed the check against last 4 poses")
                rotated_poses.append(filtered_poses[0])
            else:
                print(f"Marker ID {id[0]} FAILED the check against last 4 poses")
        else:
            rotated_poses.append(rotMat)

    avg_pose = None
    filtered_poses = []
    faulty_marker_ids = []
    if len(rotated_poses) != 0:
        avg_pose = compute_average_pose(rotated_poses)
        deviations = compute_angular_deviations(avg_pose, rotated_poses)
        filtered_poses, faulty_markers, faulty_marker_ids = detect_faulty_poses_fixed_threshold(rotated_poses, deviations, markerIds)

        print(f"{filtered_poses=}, {faulty_marker_ids=}\n")

        if len(filtered_poses) != 0:
            avg_pose = compute_average_pose(rotated_poses)

    if avg_pose is not None and len(filtered_poses) != 0:
        consequtive_failures = 0
        if len(last_4_poses) < 4:
            last_4_poses.append(avg_pose)
        else:
            last_4_poses.pop(0)
            last_4_poses.append(avg_pose)
    else:
        if len(last_4_poses) > 0:
            avg_pose = compute_average_pose(last_4_poses)
        consequtive_failures += 1
        # print(f"Consequtive Failures={consequtive_failures}")

    if consequtive_failures == 3:
        last_4_poses.clear()

    return avg_pose, rotated_poses, faulty_marker_ids

def rotation_matrix_to_euler_angles(R):
    """
    Converts a 3x3 rotation matrix to Euler angles (Yaw, Pitch, Roll) in ZYX order.
    Parameters:
        R: 3x3 numpy array, rotation matrix
    Returns:
        Yaw, Pitch, Roll in radians
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(R[1, 0], R[0, 0])  # Rotation around Z-axis
        pitch = np.arctan2(-R[2, 0], sy)    # Rotation around Y-axis
        roll = np.arctan2(R[2, 1], R[2, 2]) # Rotation around X-axis
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0

    return [yaw, pitch, roll]

def ProcessFrame(frame, file):
    global last_4_poses
    frame_copy = frame.copy()

    grayFrame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

    word_corners_total = []

    markerCorners, markerIds, rejects = aruco.detectMarkers(
        grayFrame, markerDict, parameters=paramMarkers
    )

    if markerIds is not None and markerCorners is not None:

        total_markers = range(0, markerIds.size)

        rVec_total, tVec_total = [], []

        for ids, corners, i in zip(markerIds, markerCorners, total_markers):
            if ids[0] == 15:
                markerIds = [markerIds != 15]
                continue
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(corners,
                                                        MARKER_SIZE,
                                                        cam_mat,
                                                        dist_coef,
                                                        (aruco.ARUCO_CCW_CENTER, False, cv2.SOLVEPNP_ITERATIVE))

            cv2.polylines(frame_copy, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
            corners = corners.reshape(4, 2).astype(np.float32)
            top_right = tuple(corners.astype(int)[0].ravel())

            object_points = np.array([[-MARKER_SIZE/2, MARKER_SIZE/2, 0],
                                      [MARKER_SIZE/2, MARKER_SIZE/2, 0],
                                      [MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                                      [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]], dtype=np.float32)

            reprojection_error = calculate_reprojection_error(object_points,
                                                              corners,
                                                              np.array(rVec, dtype=np.float64),
                                                              np.array(tVec, dtype=np.float64),
                                                              cam_mat,
                                                              dist_coef)

            if reprojection_error >= 2:
                corners_2 = corners[:]
                cv2.cornerSubPix(grayFrame,
                                 corners_2,
                                 (10,10),
                                 (-1,-1),
                                 (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 0.0001))
                corners_2 = corners_2.reshape(4, 2).astype(np.float32)
                new_rVec, new_tVec, _ = aruco.estimatePoseSingleMarkers([corners_2],
                                                                            MARKER_SIZE,
                                                                            cam_mat,
                                                                            dist_coef,
                                                                            (aruco.ARUCO_CCW_CENTER, False, cv2.SOLVEPNP_ITERATIVE))

                new_reprojection_error = calculate_reprojection_error(object_points,
                                                                  corners_2,
                                                                  np.array(new_rVec, dtype=np.float64),
                                                                  np.array(new_tVec, dtype=np.float64),
                                                                  cam_mat,
                                                                  dist_coef)
                if new_reprojection_error < reprojection_error:
                    print("Reprojection error improved")
                    corners = corners_2
                    rVec = new_rVec
                    tVec = new_tVec

            rVec_world, tVec_world = transform_to_world(rVec, tVec, REFERENCE_RVEC, REFERENCE_TVEC)
            world_corners = transform_corners_to_world(object_points,
                                                       rVec_world,
                                                       tVec_world)

            rVec_total.append(rVec_world)
            tVec_total.append(tVec_world)

            word_corners_total.append(world_corners)

            # Draw axes with the refined pose
            cv2.drawFrameAxes(frame_copy, cam_mat, dist_coef,  rVec, tVec, 4, 4)

            text = f"ID: {ids[0]} (x={tVec_world[0]}, y={tVec_world[1]}, z={tVec_world[2]})"
            # cv2.putText(frame_copy, text, top_right, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # plot_singular_points(tVec_total)
        avg_pose, rotated_poses, faulty_marker_ids = get_average_pose(rVec_total, markerIds)
        print(f"{avg_pose=}\n")
        
        # Convert radius from millimeters to meters (if tvec is in meters)
        center_dodecahedron, translated_points = aruco_to_dodecahedron_center(rVec_total, tVec_total, markerIds, faulty_marker_ids)

        positional_velocity, rotational_velocity = (0, 0, 0), (0, 0, 0)
        if len(last_4_poses) >= 2:
            positional_velocity, rotational_velocity =\
                compute_velocity(last_4_poses[len(last_4_poses) - 2],
                                last_4_positions[len(last_4_positions) - 2],
                                last_4_poses[len(last_4_poses) - 1],
                                last_4_positions[len(last_4_positions) - 1], TIME_PER_FRAME)

        if avg_pose is not None and center_dodecahedron is not None and len(center_dodecahedron) != 0:
            avg_pose_euler = rotation_matrix_to_euler_angles(avg_pose)

            result_string = " ".join(f"{num}" for num in center_dodecahedron[0] + avg_pose_euler + list(positional_velocity) + list(rotational_velocity))

            file.write(f"{result_string}\n")  # Use indent for pretty formatting
        
        tVec_total = [tvec.flatten().tolist() for tvec in tVec_total]

        flattened_array = np.concatenate(word_corners_total, axis=0)

        # Method 2: Using list comprehension
        # flattened_list = [point.tolist() for array in word_corners_total for point in array]
        # tVec_total += flattened_list
        # center = []
        # if center_dodecahedron is not None and translated_points is not None:
        #     tVec_total += center_dodecahedron
        #     tVec_total += translated_points

        # plot_marker_corners(word_corners_total)
        # plot_singular_points(tVec_total)

        # =========== DEMO PURPOSES =========== #

        plot_marker_rotation(avg_pose, rotated_poses)

        # =========== DEMO PURPOSES =========== #


    return frame_copy

def aruco_to_dodecahedron_center(rVec_total, tVec_total, marker_ids, faulty_marker_ids):
    global inradius_mm
    
    center_dodecahedron=[]
    for rVec, tVec, id in zip(rVec_total, tVec_total, marker_ids):
        if id[0] in faulty_marker_ids:
            continue
        # Convert rvec to rotation matrix R
        R, _ = cv2.Rodrigues(rVec)

        # Extract the normal vector (Z-axis) from the rotation matrix
        normal_vector = R[:, 2]  # Third column of R

        # Flatten tvec and normal_vector to ensure proper shapes
        tVec = tVec.flatten()
        normal_vector = normal_vector.flatten()

        # Calculate the center of the dodecahedron
        center_dodecahedron.append(tVec - normal_vector * inradius_mm)

    # Transpose the nested list
    if len(center_dodecahedron) == 0:
        columns = list(zip(*last_4_positions))
        return [sum(column) / len(column) for column in columns], None
    
    columns = list(zip(*center_dodecahedron))  # Transposes the list

    # Compute the average for each column
    average_center = [sum(column) / len(column) for column in columns]

    if len(last_4_positions) >= 4:
        last_4_positions.pop(0)

    last_4_positions.append(average_center)
    
    return [average_center], center_dodecahedron



def RunVideoCaptureDetection(vidCapturePath=None):
    print("Trying to open video capture device")
    if vidCapturePath is None:
        vc = cv2.VideoCapture(0)
    else:
        vc = cv2.VideoCapture(vidCapturePath)

    if vc.isOpened(): # try to get the first frame
        print("Video capture device opened successfully!")
        print("Initializing video capture device...")
        print("Initialization completed!")
        print("Reading first frame...")
        rval, frame = vc.read()
        print("First frame read successfully!")
    else:
        rval = False

    counter = 0
    output_dir = "refined_frames"
    output_dir_2 = "unrefined_frames"

    with open('data.txt', 'w') as file:
        while rval:
            post_process_frame = ProcessFrame(frame, file)
            rval, frame = vc.read()

            sleep(1000)

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

            DisplayFrame(post_process_frame)
        
    vc.release()

def RunTestImageDetection():
    frame = cv2.imread('geometric_rotation_images/Marker19.jpg')
    assert(frame is not None)
    cv2.namedWindow('preview', cv2.WINDOW_NORMAL)

    rval = True
    
    with open('data.txt', 'w') as file:
        while rval:
            post_process_frame = ProcessFrame(frame, file)

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

            DisplayFrame(post_process_frame)

# RunTestImageDetection()
# RunVideoCaptureDetection()
RunVideoCaptureDetection("new_dodeca_video.mp4")

cv2.destroyAllWindows()
