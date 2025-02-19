import cv2 as cv2
import numpy as np
from cv2 import aruco

markerDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
markerSize = 100 #pixels
NUM_ARUCO_MARKERS = 20

for id in range(NUM_ARUCO_MARKERS):
    markerImage = aruco.generateImageMarker(markerDict, id, markerSize)
    cv2.imwrite(f"marker_{id}.png", markerImage)
