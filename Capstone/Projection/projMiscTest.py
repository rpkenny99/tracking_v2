# from PyQt6.QtGui import QImage

# formats = [attr for attr in dir(QImage) if attr.startswith("Format_")]
# print("Available QImage Formats in PyQt6:")
# for f in formats:
#     print(f)

# from PyQt6.QtGui import QImage
# # print([attr for attr in dir(QImage) if attr.startswith("Format_")])
# print(dir(QImage))

"""
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID

# Get a list of all visible windows
window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

print("Active windows:")
for window in window_list:
    window_title = window.get("kCGWindowName", "No Name")
    owner_name = window.get("kCGWindowOwnerName", "Unknown App")
    print(f"Title: {window_title} | App: {owner_name}")

"""

# # from markerDetectionFrame import startTrackingPerspective

# import markerDetectionFrame 
# import queue
# import threading

# # import queue

# # from queue import Queue



# tracking_queue = queue.Queue()

# while True:
#     if not tracking_queue.empty():
#         # Check if there is data in the queue (non-blocking)
#         data = tracking_queue.get()
#         print(f"{data=}\n")

#     else:
#         print("Queue is empty\n")    


# import multiprocessing

# # Import the functions from the tracking script
# from markerDetectionFrame import startTrackingPerspective  # Assuming your script is saved as `tracking_script.py`

# def consume_data(queue):
#     """
#     Retrieves data from the queue and processes it.
#     """
#     while True:
#         data = queue.get()
        
#         if data is None:
#             print("No more data to process. Exiting...")
#             break

#         markerIds, rVecs, tVecs = data
#         print(f"Marker IDs: {markerIds}")
#         print(f"Rotation Vectors: {rVecs}")
#         print(f"Translation Vectors: {tVecs}")

# if __name__ == "__main__":
#     queue = multiprocessing.Queue()

#     # Start tracking in a separate process
#     tracking_process = multiprocessing.Process(target=startTrackingPerspective, args=(queue,))
#     tracking_process.start()

#     # Consume data
#     consume_data(queue)

#     # Wait for tracking process to complete
#     tracking_process.join()


"""
from threading import Thread
from queue import Queue
from markerDetectionFrame import startTrackingPerspective

def ar_consumer(raw):
    while True:
        raw_data = raw.get()
        print(f"{raw_data=}")

def main():
    raw = Queue()

    raw_tracking = Thread(target=startTrackingPerspective, args=[raw], daemon=True)
    ar = Thread(target=ar_consumer, args=[raw], daemon=True)

    raw_tracking.start()
    ar.start()

    # Run the PyQt G
    raw_tracking.join()

    ar.join()

if __name__ == "__main__":
    main()
    """
import numpy as np

# Two NumPy arrays of the same shape
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

# Element-wise multiplication of the two arrays
result = arr1 * arr2
print(result)  # Output: [10 40 90 160]