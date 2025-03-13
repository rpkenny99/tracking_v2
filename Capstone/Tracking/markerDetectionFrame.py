import csv
from asyncio import sleep
import cv2 as cv2
import numpy as np
from cv2 import aruco
import os
import math
from queue import Queue

markerDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
paramMarkers = aruco.DetectorParameters()
calib_data_path = "Capstone/Tracking/Calibration/calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

first_data = True

MARKER_SIZE = 50.8
REFERENCE_RVEC = np.array([1.66040354, -0.43797448,  0.4378116])
REFERENCE_TVEC = np.array([-132.74267261,   62.84248454,  339.17614627])

def DisplayFrame(frame):
    cv2.imshow("preview", frame)
    cv2.waitKey(1)  # Non-blocking, so AR projection starts immediately

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

def ProcessFrame_2(frame, file):

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    markerCorners, markerIds, rejects = aruco.detectMarkers(
        grayFrame, markerDict, parameters=paramMarkers
    )

    x_val, y_val, z_val, pitch_val, roll_val, yaw_val, rVec, tVec = None, None, None, None, None, None, None, None

    # NOTE: For tracking of the users line of sight, there is a chance other AruCo
    # Markers are visible. Therefore, we will have to use a unique marker ID on the glasses.
    # Such that the other markers are not picked up.
    if markerIds is not None:

        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                markerCorners, MARKER_SIZE, cam_mat, dist_coef
            )
        for i, id in enumerate(markerIds):
                cv2.drawFrameAxes(frame, cam_mat, dist_coef,  rVec[i], tVec[i], 7, 4)
                print(f"{id=}: {rVec[i]=}, {tVec[i]=}")

        frame = aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
        # Update the real-time plot
        # update_realtime_plot(pitch_val, roll_val, yaw_val)
              
    return frame, [rVec, tVec]

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

def RunVideoCaptureDetection(queue, vidCapturePath=None):
    global first_data
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

    counter = 0

    with open('Capstone/Tracking/data.txt', 'w') as file:
        while rval:
            post_process_frame, data = ProcessFrame_2(frame, file)
            if data is not None:
                if first_data:
                    # Discard first data because it is faulty
                    first_data = False
                else:
                    queue.put(data)
                # print(f"Putting: {data} into queue\n")


            rval, frame = vc.read()
            counter += 1

            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break

            DisplayFrame(post_process_frame)
        
    vc.release()

def startTrackingPerspective(queue):
    RunVideoCaptureDetection(queue)
    queue.put(None)

queue = Queue()
startTrackingPerspective(queue)

