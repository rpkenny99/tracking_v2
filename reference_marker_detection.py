import cv2
from cv2 import aruco
import numpy as np

markerDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
print(f"{markerDict=}")
paramMarkers = aruco.DetectorParameters()

calib_data_path = "calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

REFERENCE_MARKER_SIZE = 26.5 # mm

frame = cv2.imread('OriginMarker_2.jpg')
grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

frame_copy = frame.copy()

rval = True
    
while rval:
    markerCorners, markerIds, rejects = aruco.detectMarkers(
        grayFrame, markerDict, parameters=paramMarkers
    )
    if markerIds is not None and markerCorners is not None:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(markerCorners,
                                                            REFERENCE_MARKER_SIZE,
                                                            cam_mat,
                                                            dist_coef,
                                                            (aruco.ARUCO_CCW_CENTER, False, cv2.SOLVEPNP_ITERATIVE))
        total_markers = range(0, markerIds.size)
        
        for ids, corners, i in zip(markerIds, markerCorners, total_markers):
            print(f"{ids[0]=}")
            if ids[0] == 19:
                cv2.polylines(frame_copy, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
                corners = corners.reshape(4, 2).astype(np.float32)
                top_right = tuple(corners.astype(int)[0].ravel())

                print(f"{rVec[i][0]=}")
                print(f"{tVec[i][0]=}")

                cv2.drawFrameAxes(frame_copy, cam_mat, dist_coef,  rVec[i][0], tVec[i][0], 10, 4)
                cv2.putText(frame_copy, f"{ids[0]=}", top_right, cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

    new_width = 640
    new_height = 480
    resized_image = cv2.resize(frame_copy, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imshow("preview", resized_image)