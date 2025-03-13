import cv2 as cv
import cv2.aruco as aruco
import os

# --------------------------
# ChArUco board parameters
# --------------------------
# Rows: 11, Columns: 18
squares_x = 18   # number of squares along the X direction
squares_y = 11   # number of squares along the Y direction

# Keep units consistent: here, we use millimeters (mm).
square_length = 11.91  # length of each ChArUco square (mm)
marker_length = 8.67   # length of each ArUco marker inside the square (mm)

# Choose a 4x4 dictionary. For example, DICT_4X4_50.
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)

# Create the ChArUco board
charuco_board = aruco.CharucoBoard(
    (squares_x, squares_y), square_length, marker_length, aruco_dict
)

# --------------------------
# Setup image saving directory
# --------------------------
n = 0  # image counter
image_dir_path = r"Capstone/Tracking/Calibration/90FOV/images"

if not os.path.isdir(image_dir_path):
    os.makedirs(image_dir_path)
    print(f'"{image_dir_path}" Directory is created')
else:
    print(f'"{image_dir_path}" Directory already Exists.')

# --------------------------
# Function to detect a ChArUco board
# --------------------------
def detect_charuco_board(image, grayImage, board, dictionary):
    # Detect ArUco markers in the image
    corners, ids, rejected = aruco.detectMarkers(grayImage, dictionary)

    board_detected = False
    if len(corners) > 0:
        # Interpolate ChArUco corners using the detected markers
        retval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(
            corners, ids, grayImage, board
        )

        # If we get enough corners, consider the board "detected"
        if charucoCorners is not None and charucoIds is not None and len(charucoCorners) > 3:
            image = aruco.drawDetectedCornersCharuco(image, charucoCorners, charucoIds)
            print("ChArUco board detected")
            board_detected = True
        else:
            print("ChArUco board detection insufficient")
    else:
        print("No ArUco markers detected")

    return image, board_detected

# --------------------------
# Camera setup
# --------------------------
cap = cv.VideoCapture(0)

# --------------------------
# Main loop
# --------------------------
while True:
    ret_val, frame = cap.read()
    if not ret_val:
        break

    copyFrame = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect and draw the ChArUco board
    image, board_detected = detect_charuco_board(frame, gray, charuco_board, aruco_dict)

    # Display how many images have been saved
    cv.putText(
        frame,
        f"saved_img : {n}",
        (30, 40),
        cv.FONT_HERSHEY_PLAIN,
        1.4,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )

    cv.imshow("frame", frame)
    cv.imshow("copyFrame", copyFrame)

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("s") and board_detected:
        # Save the image only if a valid ChArUco board was detected
        cv.imwrite(f"{image_dir_path}/image{n}.png", copyFrame)
        print(f"saved image number {n}")
        n += 1

cap.release()
cv.destroyAllWindows()
print("Total saved Images:", n)
