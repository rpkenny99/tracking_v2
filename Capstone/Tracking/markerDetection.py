import csv
from asyncio import sleep
import cv2 as cv2
import numpy as np
from cv2 import aruco
import os
import matplotlib.pyplot as plt
import math
from queue import Queue

import pandas as pd
import itertools
from collections import deque

# Initialize global queue for velocity tracking
velocity_history = deque(maxlen=7)  # Stores last 7 translational velocities

markerDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
paramMarkers = aruco.DetectorParameters()
calib_data_path = r"Capstone/Tracking/Calibration/60FOV/calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

rotation_prev = None
translation_prev = None

consequtive_failures = 0

# dodecahedron_edge_length_mm = 12.71
dodecahedron_edge_length_mm = 13.46
inradius_mm = math.sqrt((25 + 11*math.sqrt(5))/40) * dodecahedron_edge_length_mm

length_of_rod = 52.72

first_data = True

MARKER_SIZE = 11.77
# MARKER_SIZE = 50.8
REFERENCE_RVEC = np.array([1.78278142,  1.61321357, -0.87660487])
REFERENCE_TVEC = np.array([21.28829506, -38.25557444, 465.88498356])
FPS = 30
TIME_PER_FRAME = 1/FPS

unwrap_buffers = {
    "pitch": deque(maxlen=5),
    "roll": deque(maxlen=5),
    "yaw": deque(maxlen=5),
}

x_data = []
y_data = []
z_data = []

def init_realtime_plot():
    global fig, ax, scatter_handle
    global x_data, y_data, z_data
        
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Real-time Pose (x, y, z)")

    # Initialize an empty scatter plot
    scatter_handle = ax.scatter([], [], [], c='r', marker='o')

    plt.show(block=False)

def update_realtime_plot(x, y, z):
    """
    Append the new (x, y, z) to our data arrays 
    and update the existing scatter in real time.
    """
    global x_data, y_data, z_data
    global fig, ax, scatter_handle

    # Add new pose to the data lists
    x_data.append(x)
    y_data.append(y)
    z_data.append(z)

    # Update the scatter plot data
    # We can do this by directly updating the offsets3d property:
    scatter_handle._offsets3d = (x_data, y_data, z_data)

    # Force the axis to re-calculate its limits 
    ax.set_xlim3d(min(x_data), max(x_data))
    ax.set_ylim3d(min(y_data), max(y_data))
    ax.set_zlim3d(min(z_data), max(z_data))

    # ax.set_xlim(-400, 400)
    # ax.set_ylim(-100, 100)
    # ax.set_zlim(-400, 400)

    # Redraw
    plt.draw()
    plt.pause(0.1)  # A small pause so that the figure gets updated

def DisplayFrame(frame):
    cv2.imshow("preview", frame)

def transform_to_world(rVec, tVec, rVec_origin=REFERENCE_RVEC, tVec_origin=REFERENCE_TVEC):
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



def compute_velocity(rvec_curr, tvec_curr, delta_t):
    global rotation_prev
    global translation_prev
    global velocity_history

    if rotation_prev is None or translation_prev is None:
        return (None, None, None), (None, None, None), True

    # Compute translational velocity
    v_x = (tvec_curr[0] - translation_prev[0]) / delta_t
    v_y = (tvec_curr[1] - translation_prev[1]) / delta_t
    v_z = (tvec_curr[2] - translation_prev[2]) / delta_t

    current_velocity = np.array([v_x, v_y, v_z])

    # Store velocity in history
    velocity_history.append(current_velocity)

    # Compute moving average and standard deviation if enough data is available
    if len(velocity_history) >= 7:
        velocity_array = np.array(velocity_history)
        moving_avg = np.mean(velocity_array, axis=0)
        std_dev = np.std(velocity_array, axis=0)

        # Flag data as faulty if current velocity exceeds 1.8 * standard deviation
        fault_flag = np.any(np.abs(current_velocity - moving_avg) > 1.8 * std_dev)
        # print(f"{fault_flag=}")
    else:
        fault_flag = False  # Not enough data to determine anomaly

    # Convert rvec to rotation matrices
    R_curr, _ = cv2.Rodrigues(rvec_curr)
    R_prev, _ = cv2.Rodrigues(rotation_prev)

    # Compute relative rotation matrix
    R_relative = np.dot(R_curr, R_prev.T)

    # Convert relative rotation matrix back to rotation vector
    rvec_relative, _ = cv2.Rodrigues(R_relative)

    # Compute angular velocity
    omega_x = rvec_relative[0][0] / delta_t
    omega_y = rvec_relative[1][0] / delta_t
    omega_z = rvec_relative[2][0] / delta_t

    return (v_x, v_y, v_z), (omega_x, omega_y, omega_z), fault_flag


def ensure_marker_faces_camera(rotation):
    """
    rotation: a 3x1 Rodrigues vector as typically returned by cv2.solvePnP().
              e.g. rotation[0][0], rotation[1][0], rotation[2][0].
    Modifies 'rotation' in-place if the marker is flipped.
    """
    # Convert rotation vector -> 3D numpy array
    z_axis = np.array([0, 1, 0], dtype=float)
    rvec_array = np.array([rotation[0][0], rotation[1][0], rotation[2][0]], dtype=float)

    dot_val = np.dot(rvec_array, z_axis)
    norm_rvec = np.linalg.norm(rvec_array)
    cos_theta = dot_val / (norm_rvec * 1.0)  # (||z_axis|| is 1)
    # Clamp to avoid numerical domain errors
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta = math.acos(cos_theta)
    
    # 3) Camera’s +Z is [0, 0, 1].  Check the dot product:
    #    If z_marker_cam is facing *away* from camera (dot < 0), flip 180° around local Y.
    if abs(theta) > math.radians(90):
        rotation[0][0] *= -1
        rotation[1][0] *= -1
        rotation[2][0] *= -1

    # If dot >= 0, we do nothing: it is already facing the camera.
    return rotation

def dodecahedron_center_to_iv(rotation, translation):
    """
    Computes the corrected rotation and translation of the dodecahedron marker,
    discarding points if their Z-axis is incorrect.
    
    Args:
        rotation: Rodrigues rotation vector (3x1 numpy array)
        translation: Translation vector (3x1 numpy array)
    
    Returns:
        (rotation, translation) if valid, otherwise None
    """
    # Convert Rodrigues vector to rotation matrix
    rotation, _ = cv2.Rodrigues(rotation)

    # Extract the marker's local Z-axis (third column of R_marker)
    local_z_axis = rotation[:, 2]  

    # Rotate by 144 degrees around local Z-axis
    angle_rad = np.deg2rad(72 * 2)
    R_local_z, _ = cv2.Rodrigues(local_z_axis * angle_rad)
    rotation = R_local_z @ rotation  

    # Rotate by -8 degrees around updated Z-axis
    local_z_axis = rotation[:, 2]
    angle_rad = np.deg2rad(-8)
    R_local_z, _ = cv2.Rodrigues(local_z_axis * angle_rad)
    rotation = R_local_z @ rotation  

    # Rotate by 97.2 degrees around the updated Y-axis
    local_y_axis = rotation[:, 1]
    angle_rad = np.deg2rad(106.4)
    R_local_y, _ = cv2.Rodrigues(local_y_axis * angle_rad)
    rotation = R_local_y @ rotation

    # Rotate by 97.2 degrees around the updated Y-axis
    local_x_axis = rotation[:, 0]
    angle_rad = np.deg2rad(-3)
    R_local_x, _ = cv2.Rodrigues(local_x_axis * angle_rad)
    rotation = R_local_x @ rotation

    

    translation = translation + (57.35) * rotation[:, 2].reshape(3, 1)
    translation = translation + (2) * rotation[:, 1].reshape(3, 1)

    # Rotate by -8 degrees around updated Z-axis
    local_z_axis = rotation[:, 2]
    angle_rad = np.deg2rad(7.2)
    R_local_z, _ = cv2.Rodrigues(local_z_axis * angle_rad)
    rotation = R_local_z @ rotation  

    # Rotate by -8 degrees around updated Z-axis
    local_y_axis = rotation[:, 1]
    angle_rad = np.deg2rad(-1.7)
    R_local_y, _ = cv2.Rodrigues(local_y_axis * angle_rad)
    rotation = R_local_y @ rotation  

    translation = translation + (-115.74) * rotation[:, 0].reshape(3, 1)

    # Extract the final local Z-axis after transformations
    final_z_axis = rotation[:, 2]

    # # **Check if the transformed Z-axis is positive (wrong direction)**
    if final_z_axis[2] > 0:
        print("Invalid pose: Z-axis is incorrectly positive.")
        return None, None
        # **Attempt to salvage by flipping 180° around the X-axis**
        # R_flip_x, _ = cv2.Rodrigues(np.array([np.pi, 0, 0]))  # 180° around X
        # rotation = R_flip_x @ rotation  # Apply correction

        # # Check if flipping corrected it
        # final_z_axis = rotation[:, 2]
        # if final_z_axis[2] > 0:
        #     print("Could not correct, discarding point.")
        #     return None  # Discard if still invalid

    # Convert back to Rodrigues vector
    rotation, _ = cv2.Rodrigues(rotation)

    return rotation, translation

def dodecahedron_center_to_iv_dynamic(rotation, translation, angle1, angle2):
    """
    Modified `dodecahedron_center_to_iv` function that accepts two dynamic angles.
    """
    rotation, _ = cv2.Rodrigues(rotation)

    # Extract the marker's local Z-axis (third column of R_marker)
    local_z_axis = rotation[:, 2]  # The third column is the Z-axis

    # Convert the rotation into radians
    angle_rad = np.deg2rad(72*2)

    # Create a rotation matrix about the marker's Z-axis
    R_local_z, _ = cv2.Rodrigues(local_z_axis * angle_rad)

    # Apply the rotation
    rotation = R_local_z @ rotation  # Rotate marker in its local frame

    # Extract the marker's local Z-axis (third column of R_marker)
    local_z_axis = rotation[:, 2]  # The third column is the Z-axis

    # Convert the rotation into radians
    angle_rad = np.deg2rad(angle1)

    # Create a rotation matrix about the marker's Z-axis
    R_local_z, _ = cv2.Rodrigues(local_z_axis * angle_rad)

    # Apply the rotation
    rotation = R_local_z @ rotation  # Rotate marker in its local frame

    local_y_axis = rotation[:, 1]  # The third column is the Z-axis

    # Convert the rotation into radians
    angle_rad = np.deg2rad(angle2)

        # Create a rotation matrix about the marker's Z-axis
    R_local_y, _ = cv2.Rodrigues(local_y_axis * angle_rad)

    # Apply the rotation
    rotation = R_local_y @ rotation  # Rotate marker in its local frame

    translation = translation + (inradius_mm + length_of_rod) * rotation[:, 0].reshape(3, 1)
    translation = translation + (2) * rotation[:, 1].reshape(3, 1)
    
    # Convert back to rotation vector
    rotation, _ = cv2.Rodrigues(rotation)

    return rotation, translation

def sweep_dodecahedron_transform():
    """
    Performs a nested sweep over two rotation angles, varying them from 2 to 9 in increments of 0.1,
    and records the resulting rotation and translation vectors.
    
    Saves the results into an Excel file.
    """
    # Define sweep ranges
    sweep_values = np.arange(-11, 11, 0.1)  # Sweep values from 2 to 9 with step 0.1

    # Prepare DataFrame for results
    results = []

    # Perform nested loop over both angle values
    for angle1, angle2 in itertools.product(sweep_values, repeat=2):
        
        # Modify `dodecahedron_center_to_iv` to take dynamic angles
        rotation_result, translation_result = dodecahedron_center_to_iv_dynamic(
            np.array([  [-1.5833294 ],
                        [ 1.34539269],
                        [-2.26428008]]),
            np.array([  [-1.43052298],
                        [-9.30266352],
                        [67.54010506]]),
            angle1,
            angle2
        )

        # rotation_result, translation_result = transform_to_world(rotation_result,
        #                                                          translation_result)

        # Store results
        results.append({
            "Angle1 (deg)": angle1,
            "Angle2 (deg)": angle2,
            "Rotation X": rotation_result[0][0],
            "Rotation Y": rotation_result[1][0],
            "Rotation Z": rotation_result[2][0],
            "Translation X": translation_result[0][0],
            "Translation Y": translation_result[1][0],
            "Translation Z": translation_result[2][0],
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to Excel
    excel_filename = "sweep_results.xlsx"
    df.to_excel(excel_filename, index=False)
    
    print(f"✅ Sweep complete! Results saved to {excel_filename}")


def angle_between_rodrigues_2(vec1):
    R1, _ = cv2.Rodrigues(vec1)

    x1 = R1[:, 0]
    y1 = R1[:, 1]
    z1 = R1[:, 2]

    yaw_rad = np.arctan2(x1[0], x1[1])
    yaw_deg = np.degrees(yaw_rad)

    if yaw_deg > 0:
        yaw_deg = 180 - yaw_deg
    elif yaw_deg < 0:
        yaw_deg = -180 - yaw_deg

    pitch_rad = np.arctan2(z1[1], z1[2])
    pitch_deg = np.degrees(pitch_rad)

    roll_rad = np.arctan2(y1[2], y1[1])
    roll_deg = np.degrees(roll_rad)

    return yaw_deg, pitch_deg, roll_deg

def ProcessFrame_2(frame, file):
    global board
    global consequtive_failures
    global z_data
    global paramMarkers
    global rotation_prev
    global translation_prev

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    markerCorners, markerIds, rejects = aruco.detectMarkers(
        grayFrame, markerDict, parameters=paramMarkers
    )

    x_val, y_val, z_val, pitch, roll, yaw = None, None, None, None, None, None

    if markerIds is not None:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            markerCorners, MARKER_SIZE, cam_mat, dist_coef
        )
        # for i, id in enumerate(markerIds):
        #     cv2.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)

        #     rotation, translation = transform_to_world(rVec[i], tVec[i])
        #     print(f"{id=}: {rotation=}, {translation=}")
        #     print(f"{id=}: {rVec[i]=}, {tVec[i]=}")

        success, rotation, translation = aruco.estimatePoseBoard(
            markerCorners, markerIds, board, cam_mat, dist_coef, r_vectors, t_vectors
        )

        if success:
            frame = aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

            if translation[2][0] < 0:
                translation[0][0] *= -1
                translation[1][0] *= -1
                translation[2][0] *= -1

            cv2.drawFrameAxes(frame, cam_mat, dist_coef, rotation, translation, 7, 4)

            rotation, translation = transform_to_world(rotation, translation)

            rotation, translation = dodecahedron_center_to_iv(rotation, translation)

            if rotation is None and translation is None:
                return frame, [None, None, None, None, None, None]

            _, _, fault = compute_velocity(rotation, translation, TIME_PER_FRAME)

            if fault and rotation_prev is not None and translation_prev is not None:
                return frame, [None, None, None, None, None, None]

            rotation_prev = rotation
            translation_prev = translation

            # Extract translation
            x_val = translation[0][0]
            y_val = translation[1][0]
            z_val = translation[2][0]

            # Extract rotation angles (assuming rotation is in Rodrigues form)
            pitch_val = rotation[0][0]
            roll_val = rotation[1][0]
            yaw_val = rotation[2][0]

            yaw, pitch, roll = angle_between_rodrigues_2(np.array([pitch_val, roll_val, yaw_val]))

            if pitch < 0:
                pitch = -(180 + pitch)
            else:
                pitch = 180 - pitch

            # Store values in rolling unwrap buffers
            unwrap_buffers["pitch"].append(pitch)
            unwrap_buffers["roll"].append(roll)
            unwrap_buffers["yaw"].append(yaw)

            # Apply np.unwrap when enough data is available
            if len(unwrap_buffers["pitch"]) == 5:
                pitch = np.degrees(np.unwrap(np.radians(unwrap_buffers["pitch"]), discont=np.radians(60)))[-1]
                roll = np.degrees(np.unwrap(np.radians(unwrap_buffers["roll"]), discont=np.radians(60)))[-1]
                yaw = np.degrees(np.unwrap(np.radians(unwrap_buffers["yaw"]), discont=np.radians(60)))[-1]

            # Write to text file
            result_string = f"{x_val} {y_val} {z_val} {pitch} {roll} {yaw}"
            file.write(result_string + "\n")

    return frame, [x_val, y_val, z_val, pitch, roll, yaw]

def rotation_around_y(d):
    r = np.deg2rad(d)
    return np.matrix([[np.cos(r), 0, -np.sin(r), 0], [0, 1, 0, 0], [np.sin(r), 0, np.cos(r), 0], [0, 0, 0, 1]],
                     dtype=np.float32)

def rotation_around_x(d):
    r = np.deg2rad(d)
    return np.matrix([[1, 0, 0, 0], [0, np.cos(r), -np.sin(r), 0], [0, np.sin(r), np.cos(r), 0], [0, 0, 0, 1]], 
                     dtype=np.float32)


def rotation_around_z(d):
    r = np.deg2rad(d)
    return np.matrix([[np.cos(r), np.sin(r), 0, 0], [-np.sin(r), np.cos(r), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                     dtype=np.float32)

def translation(tx, ty, tz):
    return np.matrix([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]], dtype=np.float32)

def hom2cart(p):
    return p[:-1] / p[-1]

def dodecahedron_aruco_points():
    global inradius_mm
    radius = inradius_mm
    tc = MARKER_SIZE/2 #mm
    all_aruco_points = []

    # top-left, top-right, bottom-right, bottom-left
    origin_points = np.matrix([ [-tc, -tc, 0, 1],[-tc, tc, 0, 1],[tc, tc, 0, 1],[tc, -tc, 0, 1]], dtype=np.float32).T

    #ids = [5, 1, 2, 3, 4]
    sticker_rotations = [180, 90, -90, -90, 0]
    pentagon_edge_to_marker = [2.18, 2.04, 2.00, 2.14, 1.62]
    marker_edge_to_marker_center = MARKER_SIZE/2
    pentagon_edge_to_pentagon_center = math.tan(math.radians(54)) * dodecahedron_edge_length_mm/2

    adjustment = [0,0,0,0,0]
    adjustment[0] = translation((pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[0])), 0, 0)
    adjustment[1] = translation(-(pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[1])), 0, 0)
    adjustment[2] = translation(0, -(pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[2])), 0)
    adjustment[3] = translation(0, -(pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[3])), 0)
    adjustment[4] = translation(0, (pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[3])), 0)

    for i, rot in enumerate([0, 4, 3, 2, 1]):
        aruco_corners = rotation_around_z(72 * rot) * rotation_around_y(116.565) *\
            rotation_around_z(sticker_rotations[rot]) * \
                translation(0, 0, -radius) * rotation_around_y(180) * adjustment[i] * origin_points
        all_aruco_points.append(hom2cart(aruco_corners).T)
    

    pentagon_edge_to_marker = [2.13, 2.14, 2.78, 2.79, 2.45]
    adjustment[0] = translation((pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[0])), 0, 0)
    adjustment[1] = translation((pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[1])), 0, 0)
    adjustment[2] = translation(-(pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[2])), 0, 0)
    adjustment[3] = translation(0, (pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[3])), 0)
    adjustment[4] = translation(-(pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + pentagon_edge_to_marker[4])), 0, 0)

    # ids =[6, 14, 9, 8, 7]
    sticker_rotations = [180, 0, 0, 180, 90]
    for i, rot in enumerate([2, 1, 0, 4, 3]):
        aruco_corners = rotation_around_z(72*rot) * rotation_around_y(116.565) * \
                        rotation_around_z(180) * rotation_around_z(sticker_rotations[rot]) * translation(0, 0, radius) * adjustment[i] * origin_points
        all_aruco_points.append(hom2cart(aruco_corners).T)

    # # top marker0
    aruco_corners = rotation_around_z(90) * translation(0, 0, radius) * translation(0, (pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + 2.27)), 0) * origin_points
    all_aruco_points.append(hom2cart(aruco_corners).T)
    # # bottom marker19
    aruco_corners = rotation_around_z(-72) * translation(0, 0, -radius) * translation(0, (pentagon_edge_to_pentagon_center - (marker_edge_to_marker_center + 2.73)), 0) * rotation_around_y(180) * origin_points
    all_aruco_points.append(hom2cart(aruco_corners).T)
  
    return all_aruco_points

def RunVideoCaptureDetection(queue, tracking_ready, vidCapturePath=None):
    print("Trying to open video capture device")
    
    # init_realtime_plot()
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

    tracking_ready.put(1)

    with open('Capstone/Tracking/data.txt', 'w') as file:
        while rval:
            post_process_frame, data = ProcessFrame_2(frame, file)
            if data[0] != None:
                queue.put(data)
                # print(f"Putting: {data} into queue\n")


            rval, frame = vc.read()

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

            DisplayFrame(post_process_frame)
        
    vc.release()

def RunTestImageDetection():
    frame = cv2.imread('Capstone/Tracking/marker_19_faulty.jpg')
    assert(frame is not None)
    cv2.namedWindow('preview', cv2.WINDOW_NORMAL)

    rval = True
    
    # Define output CSV file
    output_csv = 'data.csv'
    fieldnames = [
        'Center_X', 'Center_Y', 'Center_Z',
        'Yaw', 'Pitch', 'Roll',
        'Positional_Velocity_X', 'Positional_Velocity_Y', 'Positional_Velocity_Z',
        'Rotational_Velocity_X', 'Rotational_Velocity_Y', 'Rotational_Velocity_Z'
    ]

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row

        while rval:
            # Process the current frame
            post_process_frame = ProcessFrame_2(frame, writer)

            sleep(1000)  # Artificial delay

            key = cv2.waitKey(20)
            if key == 27:  # Exit on ESC
                break

            DisplayFrame(post_process_frame)

def setupArucoBoard():
    points = dodecahedron_aruco_points()
    # plot_aruco_board()
    ids = np.array([[5], [1], [2], [3], [4], [9], [14], [6], [7], [8], [0], [19]])
    return aruco.Board(points, markerDict, ids)

def plot_aruco_board():
    ids = np.array([[5], [1], [2], [3], [4], [9], [14], [6], [7], [8], [0], [19]])
    aruco_points = dodecahedron_aruco_points()

    # Plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # To store all the points for axis limits
    all_x, all_y, all_z = [], [], []

    # Assuming the corners are ordered as top-left, top-right, bottom-right, bottom-left for each marker
    for idx, corners in enumerate(aruco_points):
        # Ensure corners are converted to NumPy arrays
        corners = np.array(corners)

        # Extract x, y, z coordinates for the current marker's corners
        x = corners[:, 0]
        y = corners[:, 1]
        z = corners[:, 2]

        print(f"{x=}, {y=}, {z=}\n")

        # Collect all points for axis limits
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)

        # # Plot the corners
        # ax.scatter(x, y, z, color='black', zorder=1)

        # Connect the corners to form a closed quadrilateral with grey outlines
        ax.plot(
            np.concatenate([x, [x[0]]]),
            np.concatenate([y, [y[0]]]),
            np.concatenate([z, [z[0]]]),
            '-o', color='grey', zorder=1
        )

        # Plot single points for top-right and top-left corners
        top_right = corners[1]  # Assuming index 1 is the top-right corner
        top_left = corners[0]   # Assuming index 0 is the top-left corner
        
        if idx == 0:
            ax.scatter(top_right[0], top_right[1], top_right[2], color='red', label='top-right', zorder=0)
            ax.scatter(top_left[0], top_left[1], top_left[2], color='blue', label='top-left',zorder=0)
        else:
            ax.scatter(top_right[0], top_right[1], top_right[2], color='red', zorder=0)
            ax.scatter(top_left[0], top_left[1], top_left[2], color='blue', zorder=0)

        # Add a label with the marker's ID at the center of the marker
        marker_id = ids[idx][0]  # Get the corresponding marker ID
        center_x = np.mean(x)
        center_y = np.mean(y)
        center_z = np.mean(z)
        ax.text(center_x, center_y, center_z, f'{marker_id}', color='black', fontsize=15)

    # Set equal aspect ratio for X, Y, Z
    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)
    ) / 2.0

    mid_x = (max(all_x) + min(all_x)) * 0.5
    mid_y = (max(all_y) + min(all_y)) * 0.5
    mid_z = (max(all_z) + min(all_z)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('3D Plot of ArUco Marker Corners with IDs')
    plt.show()
    plt.pause(40)  # A small pause so that the figure gets updated


def startTracking(queue, tracking_ready):
    global board
        
    board = setupArucoBoard()
    # plot_aruco_board()
    RunVideoCaptureDetection(queue, tracking_ready)
    # RunTestImageDetection()
    
    queue.put(None)

if __name__ == "__main__":
    queue, tracking_ready = Queue(), Queue()
    startTracking(queue, tracking_ready)

# sweep_dodecahedron_transform()

