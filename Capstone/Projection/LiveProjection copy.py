import sys
# import os
# # sys.path.append(os.path.join("Capstone", "Tracking"))
# sys.path.append(os.path.join("Capstone", "Feedback"))
# sys.path.append(os.path.join("Capstone", "SignalProcessing"))

import cv2
import numpy as np
import threading
import time
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
# from Feedback.updated_feeback import MainApplication  # Import your UI class
from Projection.feedback3 import FeedbackUI  # Import your UI class
from Filter.Filter_Graphs import process_file_2

from queue import Queue


from threading import Thread
from queue import Queue
from Projection.markerDetectionFrame import startTrackingPerspective


def qpixmap_to_numpy(pixmap):
    """
    Convert a QPixmap to a NumPy array (BGR format for OpenCV).
    """
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

def apply_projection_transform(image, angle_deg=-25):
    """
    Applies a perspective transformation to simulate a projection plane
    rotated by angle_deg relative to the source screen.
    """
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle_deg)
    new_top_width = w * np.cos(angle_rad)
    offset = (w - new_top_width) / 2

    src_pts = np.float32([[0, 0],
                          [w, 0],
                          [w, h],
                          [0, h]])
    dst_pts = np.float32([[offset, 0],
                          [w - offset, 0],
                          [w, h],
                          [0, h]])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def compute_viewer_offset(viewer_rvec, viewer_tvec, image_width, image_height, f=500):
    """
    Computes the offset required so that the image center aligns with the
    viewer's line of sight. A simple pinhole camera model is assumed with focal length f.
    
    Parameters:
        viewer_rvec (np.array): Rotation vector (unused in this simple model)
        viewer_tvec (np.array): Translation vector (shape (1,3) or (3,))
        image_width (int): Width of the display canvas in pixels
        image_height (int): Height of the display canvas in pixels
        f (int): Assumed focal length in pixel units
        
    Returns:
        offset_x, offset_y (int): Offsets in pixels to adjust the image center.
    """
    # Extract X, Y, Z from the translation vector
    X, Y, Z = viewer_tvec.flatten()
    # Project the origin using the pinhole camera model
    proj_x = f * X / Z + image_width / 2
    proj_y = f * Y / Z + image_height / 2
    # Compute the offset needed to bring the canvas center to this projected point
    offset_x = proj_x - (image_width / 2)
    offset_y = proj_y - (image_height / 2)
    return int(offset_x), int(offset_y)

def update_display(feedback_ui, process_Queue, angle_deg=-25, scale_factor=0.7, offset_x=0, offset_y=0 ):
    """
    Capture the updated UI, flip it for reflection, apply the projection transform,
    resize the result, and then display it with an offset.
    
    If viewer_rvec and viewer_tvec are provided, the offset is computed so that
    the image center aligns with the viewer's line of sight.
    """

    viewer_rvec, viewer_tvec= process_Queue.get() #Cange for Queue
    # print(f"Viewer Rotation Vector: {viewer_tvec}")

    # Capture the current UI as a QPixmap
    pixmap = feedback_ui.grab()
    # Convert to a NumPy array (BGR format)
    original_image = qpixmap_to_numpy(pixmap)
    
    # Flip the image vertically (simulate reflection)
    flipped_image = cv2.flip(original_image, 0)
    
    # Apply the projection transformation
    transformed_image = apply_projection_transform(flipped_image, angle_deg=angle_deg)

    # Get the dimensions of the transformed image (our display canvas)
    window_height, window_width = transformed_image.shape[:2]

    # Scale down the image
    new_width = int(window_width * scale_factor)
    new_height = int(window_height * scale_factor)
    resized_image = cv2.resize(transformed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # If viewer data is provided, compute offsets based on the viewer's perspective
    viewer_rvec = [1,1,1] #Chanhe when using rVec
    if viewer_rvec is not None and viewer_tvec is not None:

        print(f"Viewer Rotation Vector: {viewer_rvec}")
        print(f"Viewer Translation Vector: {viewer_tvec}")
        offset_x, offset_y = compute_viewer_offset(viewer_rvec, viewer_tvec, window_width, window_height)

    # Create a blank canvas with a black background
    display_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)

    # Calculate the starting position (centered plus offsets)
    start_x = (window_width - new_width) // 2 + offset_x
    start_y = (window_height - new_height) // 2 + offset_y

    # Clip the region if the resized image extends beyond the canvas
    # Determine the ROI in the canvas
    x1 = max(start_x, 0)
    y1 = max(start_y, 0)
    x2 = min(start_x + new_width, window_width)
    y2 = min(start_y + new_height, window_height)
    
    #  What does this do?????
    #  Determine the corresponding ROI in the resized_image
    roi_x1 = 0 if start_x >= 0 else -start_x
    roi_y1 = 0 if start_y >= 0 else -start_y
    roi_x2 = roi_x1 + (x2 - x1)
    roi_y2 = roi_y1 + (y2 - y1)

    # Overlay the valid region of the resized image onto the canvas
    display_image[y1:y2, x1:x2] = resized_image[roi_y1:roi_y2, roi_x1:roi_x2]

    # Display the result using OpenCV
    cv2.imshow("Projected UI", display_image)
    cv2.waitKey(1)  # Brief delay for OpenCV event processing

"""
def ar_consumer(raw):
    while True:
        raw_data = raw.get()
        print(f"{raw_data=}")

def processTracking(raw):
    while True:
        raw_data = raw.get()
"""

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

        r_data, t_data = process_tracking_data(marker_ids_list, r_vec_list, t_vec_list)
        
        process_Queue.put([r_data, t_data]) 


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
        # elif marker_id == 30:
        #     tvec_reference = np.array(t_vec_list[i], dtype=float)

    # Only do the math if we have all three
    if tvec_27 is not None and tvec_28 is not None :
        view_tvec = (tvec_27 + tvec_28) / 2.0
        view_rvec = None  # Not used in this example
        # viewer_perspective = view_tvec - tvec_reference
        return view_rvec, view_tvec
    else:
        # We don't have all needed markers; return None or handle as needed
        return None, None


        
def start(sig_processed):

    """ 
    app_to_signal_processing = Queue()
    sig_processed = Queue()
    angle_range_queue = Queue()
    direction_intruction_queue = Queue()

    # filterQueue = Queue()
    # filter = Thread(target=process_file_2, args=[raw, filtered, simulation_running_queue, file_lock], daemon=True)
    """

    # (Optional) push any initial data into angle_range_queue if needed:
    # angle_range_queue.put((expert_pitch, some_value, expert_yaw, ...))

    # Create the QApplication (if you have not already)
    raw = Queue()
    process_Queue = Queue()


    
    # app = QApplication(sys.argv)

    # app_to_siignal_processing = Queue()
    # angle_range_queue = Queue()
    # direction_intruction_queue = Queue()

    raw_tracking = Thread(target=startTrackingPerspective, args=[raw], daemon=True)
    raw_tracking.start()
    processedData = Thread(target=ar_consumer, args=[raw,process_Queue], daemon=True)
    processedData.start()

    # filterQueue = Queue()
    # filterQueue = Thread(target=process_file_2, args=[process_Queue, filterQueue, Queue(), Queue()], daemon=True)
    # Create and run the MainApplication, which handles
    # all the user dialogs (Intro, PickVein, etc.)
    # feedback_ui = FeedbackUI()
    
    # MainApplication(self,
    #     sig_processed,
    #     app_to_siignal_processing,
    #     angle_range_queue,
    #     direction_intruction_queue
    # )
    # MainApplication.run()

    # tracking_data = ar_consumer(raw)


    app = QApplication(sys.argv)

    selected_vein = "Left Vein"
    selected_point = "Point A"
    work_queue = Queue()  # Provide an empty queue if needed

    feedback_ui = FeedbackUI(selected_vein, selected_point, work_queue=sig_processed)
    feedback_ui.show()
    # # Supply required parameters and a dummy queue for FeedbackUI
    # selected_vein = "Left Vein"
    # selected_point = "Point A"
    work_queue = Queue()  # Provide an empty queue if needed

    feedback_ui.show()

    """"
    # For demonstration, here is some dummy viewer perspective data:
    # This data might be received from an external source.
    viewer_rvec = np.array([[-0.0681944 ,  2.67529694,  0.22829397]])
    viewer_tvec = np.array([[162.8271214 ,  70.32294188, 169.37338088]])"
    
    viewer_rvec = None
    viewer_tvec = None
    """
    
    
    # Set up a QTimer to update the projection capture periodically (every 100 ms)
    # When viewer_rvec and viewer_tvec are provided, the image center will align with the viewer's line of sight.
    timer = QTimer()
    # timer.timeout.connect(lambda: update_display(feedback_ui, angle_deg=25, viewer_rvec=viewer_rvec, viewer_tvec=viewer_tvec))
    timer.timeout.connect(lambda: update_display(feedback_ui, process_Queue, angle_deg=25))
    timer.start(100)  # Update every 100 milliseconds

    sys.exit(app.exec())

if __name__ == "__main__":

    # sig_processed = Queue()
    # app_to_signal_processing = Queue()
    # angle_range_queue = Queue()
    # direction_intruction_queue = Queue()
    start(sig_processed=Queue())

    # main_app = MainApplication(sig_processed, app_to_signal_processing, angle_range_queue, direction_instruction_queue)
    # main_app.run()