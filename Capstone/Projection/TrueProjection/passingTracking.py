import threading
import queue
import numpy as np

###############################################################################
# 1) A dictionary to hold poses (rvec/tvec) for each tracked object.
###############################################################################
positions = {
    "user":    {"rvec": None, "tvec": None},
    "glass":   {"rvec": None, "tvec": None},
    "monitor": {"rvec": None, "tvec": None},
    "object":  {"rvec": None, "tvec": None},
    # Add more if you have more markers
}


###############################################################################
# 2) Function that receives (marker_id, rvec, tvec) and stores them.
#    You decide how marker_id maps to 'user', 'glass', etc.
###############################################################################
def store_pose(marker_id, rvec, tvec):
    """
    Store the (rvec, tvec) in the positions dictionary,
    matching each marker ID to the correct key.
    """
    # Example mapping:
    if marker_id == 127:
        positions["user"]["rvec"] = rvec
        positions["user"]["tvec"] = tvec
    elif marker_id == 128:
        positions["glass"]["rvec"] = rvec
        positions["glass"]["tvec"] = tvec
    elif marker_id == 129:
        positions["monitor"]["rvec"] = rvec
        positions["monitor"]["tvec"] = tvec
    elif marker_id == 130:
        positions["object"]["rvec"] = rvec
        positions["object"]["tvec"] = tvec
    else:
        # You can handle unknown marker IDs here
        print(f"Warning: Unrecognized marker_id={marker_id}")


###############################################################################
# 3) Worker thread function that pulls data off the queue
#    and calls store_pose() to update global variables.
###############################################################################
def tracking_thread_func(data_queue):
    while True:
        data_item = data_queue.get()
        if data_item is None:
            # A sentinel value (None) to tell the thread to stop
            break
        
        # data_item is expected to be a tuple: (marker_id, rvec, tvec)
        marker_id, rvec, tvec = data_item
        
        # Store in the global positions dictionary
        store_pose(marker_id, rvec, tvec)
        
        data_queue.task_done()


###############################################################################
# 4) Example main loop to show how you might push data to the queue
###############################################################################
def main():
    # Create the queue for passing data to the thread
    data_queue = queue.Queue()
    
    # Start the tracking thread
    t = threading.Thread(target=tracking_thread_func, args=(data_queue,), daemon=True)
    t.start()
    
    # -------------------------------------------------------------------------
    # SIMULATED EXAMPLE:
    # We'll pretend we received two sets of data from the tracker
    # in real usage, you'd be reading from your actual tracking source
    # and parsing the raw strings into numeric arrays (marker_id, rvec, tvec).
    # -------------------------------------------------------------------------
    
    # Example 1
    marker_id_1 = 127
    rvec_1 = np.array([3.87060466, -0.80475975, -0.38239898], dtype=np.float32)
    tvec_1 = np.array([-27.4, 38.48716711, 208.1231688],    dtype=np.float32)
    data_queue.put((marker_id_1, rvec_1, tvec_1))
    
    # Example 2
    marker_id_2 = 128
    rvec_2 = np.array([3.8625888, -0.8536817, -0.4712357], dtype=np.float32)
    tvec_2 = np.array([-27.5, 42.333147, 217.309219],      dtype=np.float32)
    data_queue.put((marker_id_2, rvec_2, tvec_2))
    
    # Wait for all items to be processed
    data_queue.join()
    
    # Stop the thread
    data_queue.put(None)
    t.join()
    
    # -------------------------------------------------------------------------
    # At this point, positions dictionary should have updated data.
    # You can now use positions["user"]["rvec"], positions["user"]["tvec"], etc.
    # in your AR display code or anywhere else.
    # -------------------------------------------------------------------------
    print("Final poses stored in 'positions':")
    for key, val in positions.items():
        print(f"{key} => rvec={val['rvec']}, tvec={val['tvec']}")

if __name__ == "__main__":
    main()
