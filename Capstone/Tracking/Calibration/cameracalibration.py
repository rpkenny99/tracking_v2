import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import os

# -------------------------------------------------------
# 1. Define your ChArUco board parameters
# -------------------------------------------------------
# For example, if your board has 11 rows and 18 columns of squares
squares_x = 18   # number of squares in the horizontal direction
squares_y = 11   # number of squares in the vertical direction

# Physical dimensions in millimeters
square_length = 11.58  # size of each square (mm)
marker_length = 8.67   # size of each marker (mm)

# Choose a 4x4 dictionary. For example, DICT_4X4_50.
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

# Create the ChArUco board object
charuco_board = aruco.CharucoBoard(
    (squares_x, squares_y), square_length, marker_length, aruco_dict
)

# -------------------------------------------------------
# 2. Setup directories
# -------------------------------------------------------
calib_data_path = "Capstone/Tracking/Calibration/78FOV/calib_data"
if not os.path.isdir(calib_data_path):
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')
else:
    print(f'"{calib_data_path}" Directory already Exists.')

image_dir_path = "Capstone/Tracking/Calibration/78FOV/images"
files = os.listdir(image_dir_path)

# -------------------------------------------------------
# 3. Prepare lists to collect corners/IDs from all images
# -------------------------------------------------------
all_corners = []  # will store the 2D corners for each image
all_ids = []      # will store the corner IDs for each image
image_size = None # will store the (width, height) of images

# -------------------------------------------------------
# 4. Loop over all ChArUco images to detect markers/corners
# -------------------------------------------------------
for file in files:
    image_path = os.path.join(image_dir_path, file)
    image = cv.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}, skipping...")
        continue

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if image_size is None:
        # Set image_size from the first valid image
        image_size = gray.shape[::-1]  # (width, height)

    # 4A. Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    # 4B. If markers detected, interpolate to find ChArUco corners
    if ids is not None and len(ids) > 0:
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=charuco_board
        )

        # 4C. Store corners/IDs if we have a valid detection
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

            print(f"ChArUco detected in {file}: {len(charuco_corners)} corners")
        else:
            print(f"ChArUco corners not sufficient in {file}")
    else:
        print(f"No ArUco markers detected in {file}")

# -------------------------------------------------------
# 5. Calibrate the camera using all collected charuco data
# -------------------------------------------------------
if len(all_corners) == 0:
    print("No ChArUco corners collected. Calibration cannot proceed.")
    exit(0)

# The calibrateCameraCharuco function does the heavy lifting:
# It uses all the Charuco corners/IDs and known board geometry
# to compute camera intrinsics + distortion.
try:
    print("\nRunning camera calibration...")
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print("Calibration successful!")
    print("Reprojection error:", retval)
except cv.error as e:
    print("Calibration failed:", e)
    exit(0)

# -------------------------------------------------------
# 6. Save the calibration data to .npz
# -------------------------------------------------------
print("\nSaving calibration data to disk...")
np.savez(
    f"{calib_data_path}/MultiMatrix",
    camMatrix=cameraMatrix,
    distCoef=distCoeffs,
    rVector=rvecs,
    tVector=tvecs,
)

print("Camera matrix:\n", cameraMatrix)
print("Distortion coefficients:\n", distCoeffs)
print("rvecs (example):", rvecs[0] if rvecs else None)
print("tvecs (example):", tvecs[0] if tvecs else None)

print("\nCalibration data saved to", f"{calib_data_path}/MultiMatrix.npz")

# -------------------------------------------------------
# 7. (Optional) Demonstrate loading the data
# -------------------------------------------------------
data = np.load(f"{calib_data_path}/MultiMatrix.npz")
loadedCamMatrix = data["camMatrix"]
loadedDist = data["distCoef"]

print("\nLoaded calibration data successfully!")
print("Loaded camera matrix:\n", loadedCamMatrix)
print("Loaded distortion:\n", loadedDist)
