import csv
from asyncio import sleep
import cv2 as cv2
import numpy as np
from cv2 import aruco
import os
# import matplotlib.pyplot as plt
import math
from queue import Queue

markerDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
paramMarkers = aruco.DetectorParameters()
calib_data_path = "Capstone/Tracking/calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 46
REFERENCE_RVEC = np.array([   1.23632516, -1.38610666,  1.08522845 ])
REFERENCE_TVEC = np.array([ -15.74993467, 118.79749162, 365.1193848 ])
FPS = 30
TIME_PER_FRAME = 1/FPS


def DisplayFrame(frame):
    cv2.imshow("preview", frame)

def ProcessFrame_2(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_detector = cv2.aruco.ArucoDetector(markerDict, paramMarkers)

    markerCorners, markerIds, rejects = aruco_detector.detectMarkers(grayFrame)

    if markerIds is not None:

        rVec, tVec, success = cv2.aruco.estimatePoseSingleMarkers(
                markerCorners, MARKER_SIZE, cam_mat, dist_coef
            )
        
        frame = aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        for i, _ in enumerate(markerIds):
            cv2.drawFrameAxes(frame, cam_mat, dist_coef,  rVec[i], tVec[i], 10, 4)
        
        # Extract translation and rotation
        print(f"{rVec=}, {tVec=}\n")
              
    return frame


def RunTestImageDetection():
    frame = cv2.imread('Capstone/Vein/Trial2/WIN_20250221_17_12_21_Pro.jpg')
    assert(frame is not None)
    cv2.namedWindow('preview', cv2.WINDOW_NORMAL)

    rval = True
    aruco_detector = cv2.aruco.ArucoDetector(markerDict, paramMarkers)

    while rval:
        # Process the current frame
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        markerCorners, markerIds, rejects = aruco_detector.detectMarkers(grayFrame)

        print(f"{markerIds=}")

        if markerIds is not None:

            rVec, tVec, success = cv2.aruco.estimatePoseSingleMarkers(
                    markerCorners, MARKER_SIZE, cam_mat, dist_coef
                )
            
            frame = aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            for i, _ in enumerate(markerIds):
                cv2.drawFrameAxes(frame, cam_mat, dist_coef,  rVec[i], tVec[i], 10, 4)
            
            # Extract translation and rotation
            print(f"{rVec=}, {tVec=}\n")

        key = cv2.waitKey(20)
        if key == 27:  # Exit on ESC
            break

        DisplayFrame(frame)

def RunVideoCaptureDetection(vidCapturePath=None):
    print("Trying to open video capture device")
    
    # init_realtime_plot()
    if vidCapturePath is None:
        vc = cv2.VideoCapture(1)
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

    while rval:
        post_process_frame = ProcessFrame_2(frame)

        rval, frame = vc.read()
        counter += 1

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

        DisplayFrame(post_process_frame)
        
    vc.release()


def startTracking():
    RunVideoCaptureDetection()
    # RunTestImageDetection()

startTracking()