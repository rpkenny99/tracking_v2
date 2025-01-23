import cv2 as cv2
import numpy as np
from cv2 import aruco


markerDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

NUM_FRAMES = 10

cv2.namedWindow("preview")
vc = cv2.VideoCapture(2)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:

    # Display
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")