import sys
import threading
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
import time
from queue import Queue
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from Tracking.markerDetectionFrame import startTrackingPerspective

WORLD_COORD_TO_ARM = np.array([81.66, 97.77 , 0.0], dtype=float) # mm right, up
Original_marker = np.array([0, 0, 0], dtype=float)  # Placeholder for the original marker position

def ar_consumer(raw, process_Queue):
    # This function acts as the consumer.
    # It continuously retrieves data from the queue, expecting each item to be a tuple or list with three elements.
    while True:
        raw_data = raw.get()            # Wait for the next item from the queue.
        
        # Unpack the data into three parts:
    
    # while True :

        marker_ids_list = []
        r_vec_list = []
        t_vec_list = []

        marker_ids, r_vec, t_vec = raw_data

        if marker_ids is not None:

            # marker_ids, r_vec, t_vec = raw_data

            # Convert the marker_ids array to a Python list
            marker_ids_list = np.array(marker_ids).flatten().tolist()
            # print("Marker IDs:", marker_ids_list)

            # Convert rotation vectors to a Python list
            # Often r_vec has shape (N,1,3), so you may want to flatten it
            r_vec_list = np.array(r_vec).reshape(-1,3).tolist()  
            # print("Rotation Vectors:", r_vec_list)
                    
            t_vec_list = np.array(t_vec).reshape(-1,3).tolist()
            # print("Translation Vectors:", t_vec_list)    

            # return marker_ids_list, r_vec_list, t_vec_list

        r_data, t_data, LOS_data = process_tracking_data(marker_ids_list, r_vec_list, t_vec_list)


        veinPOS, veinPOSX, veinPOSY, veinPOSZ = vein_UI(LOS_data)
     
        # process_Queue.put([r_data, t_data]) 
        # viewerLos.put(LOS_data)
        process_Queue.put([veinPOSX]) #If callibration for veritical starting point it done in vein_UI add veinPOSY
        
def process_tracking_data(marker_ids_list, r_vec_list, t_vec_list):
    # marker_ids_list is presumably something like [27, 28, 30].
    # r_vec_list and t_vec_list have the same length, e.g., each index i corresponds to marker_ids_list[i].

    # Initialize to None so we can check them
    tvec_27 = None
    tvec_28 = None
    # tvec_reference = None

    # If marker_ids_list has multiple IDs, loop through them
    for i, marker_id in enumerate(marker_ids_list):
        # The translation vector for this marker is t_vec_list[i]
        # The rotation vector for this marker is r_vec_list[i]
        if marker_id == 27:
            tvec_27 = np.array(t_vec_list[i], dtype=float)  # convert to NumPy array
        elif marker_id == 28:
            tvec_28 = np.array(t_vec_list[i], dtype=float)
        elif marker_id == 30:
            tvec_reference = np.array(t_vec_list[i], dtype=float)
            arm_reference = tvec_reference + WORLD_COORD_TO_ARM

    # Only do the math if we have all three
    if tvec_27 is not None and tvec_28 is not None and tvec_reference is not None:
        view_tvec = (tvec_27 + tvec_28) / 2.0
        view_rvec = None  # Not used in this example
        viewer_perspective = view_tvec - arm_reference
        return view_rvec, view_tvec, viewer_perspective
    
    else:
        # We don't have all needed markers; return None or handle as needed
        return None, None, None

def vein_UI(LOS_data):

    Max_X = 1920.0
    centreX = Max_X / 2.0
    Max_y = 1080.0
    centreY = Max_y / 2.0

    """Replace with actual physical monitor dimentions data"""
    Monitor_X = 1920.0
    Monitor_Y = 1080.0
    Monitor_X_Centre = Monitor_X / 2.0
    Monitor_Y_Centre = Monitor_Y / 2.0
    planeCentre = np.array([Monitor_X_Centre, 0.0, 0.0], dtype=float)
    """Check waht marker detection calls X and Y axis"""
    """If there's time add 25 degrees offset to account for viewer starting positio on vertical length of screen"""
    """if vertical start point is caliberated use commented line below"""
    # planeCentre = np.array([Monitor_X_Centre, Monitor_Y_Centre, 0.0], dtype=float)
    


    screenScaleX = Max_X / Monitor_X
    screenScaleY = Max_y / Monitor_Y
    screenScale = np.array([screenScaleX, screenScaleY, 0.0], dtype=float)

    screeCentreX = Monitor_X_Centre * screenScaleX
    screenCentreY = Monitor_Y_Centre * screenScaleY
    screenCentre = np.array([screeCentreX, screenCentreY, 0], dtype=float) 
    
    
    
    Mag_PlanetoPhantom = 300  #mm 
    """Change to actual data """
    Mag_LOS = np.linalg.norm(LOS_data)
    print("Magnitude of LOS:", Mag_LOS)

    Mag_UsertoPlane = Mag_LOS - Mag_PlanetoPhantom

    scaleFactor = Mag_UsertoPlane / Mag_LOS

    scaleVeinDisplace = Mag_PlanetoPhantom - Mag_LOS

    # Position of the vein in the plane
    veinPosition = LOS_data * scaleVeinDisplace 

    veinPositionX = veinPosition[0] 
    veinPositionY = veinPosition[1]
    veinPositionZ = veinPosition[2]

    """Assuming top left of plane is 0,0 and bottom right is 1920,1080"""
    planeAbs = veinPosition + planeCentre
    veininUI = planeAbs * screenScale

    veininUIX = veininUI[0]
    veininUIY = veininUI[1]
    veininUIZ = veininUI[2]
    print("Vein Position in UI Coordinates:", veininUIX, veininUIY, veininUIZ)

     # return veinPosition, veinPositionX, veinPositionY, veinPositionZ
    return veininUI, veininUIX, veininUIY, veininUIZ

    
    

    """magnitude = np.linalg.norm(vector)
    print("Magnitude:", magnitude)"""



    

def main():

    #I'm actively using rawTrack and process_Queue for now. These two achieve minimal functionality. But the data needs to be filtered.
    rawTrack = Queue()
    process_Queue = Queue()
    """
    calcTrack = Queue()
    viewerLos = Queue()
    veinPosition = Queue()

    
    
    Los_tracking = threading.Thread(target=ar_consumer, args=[rawTrack, viewerLos], daemon=True)
    Los_tracking.start()
    vein_Return = threading.Thread(target= vein_UI, args=[viewerLos, veinPosition], daemon=True)
    """
    raw_tracking = threading.Thread(target=startTrackingPerspective, args=[rawTrack], daemon=True)
    raw_tracking.start()
    calc_tracking = threading.Thread(target=ar_consumer, args=[rawTrack, process_Queue], daemon=True)
    calc_tracking.start()
    """Add data filtering 'get' here"""


if __name__ == "__main__":
    '''
    app = QApplication(sys.argv)
    main_app = MainApplication()
    main_app.show()
    sys.exit(app.exec())
    '''