import random
import threading
import multiprocessing
import logging
from threading import Thread
from queue import Queue
from Tracking.markerDetection import startTracking
from Filter.Filter_Graphs import process_file_3
from SignalProcessing.signal_processing import sig_processing, load_data
import time
import glob
import numpy as np
import os
logging.basicConfig(format='%(levelname)s - %(asctime)s.%(msecs)03d: %(message)s',datefmt='%H:%M:%S', level=logging.DEBUG)

def display(msg):
    threadname = threading.current_thread().name
    processname = multiprocessing.current_process().name
    logging.info(f'{processname}\{threadname}: {msg}')

def create_work(queue, finished, max):
    finished.put(False)
    for x in range(max):
        v = random.randint(1, 100)
        queue.put(v)
        display(f'Producing: {x}: {v}')
    finished.put(True)
    display('Finished')

def perform_work(work):
    while True:
        item = work.get()               # blocking get
        if item is None:               # <-- sentinel
            break
        # Otherwise, process item
        display(f"Consumed: {item}")



def load_filtered_data(filepath):
    try:
        # Try comma delimiter first
        data = np.loadtxt(filepath, delimiter=',')
    except Exception:
        try:
            # Try space delimiter fallback
            data = np.loadtxt(filepath)
        except Exception as e:
            raise ValueError(f"Could not parse {filepath}: {e}")

    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[1] != 6:
        raise ValueError(f"Invalid shape {data.shape} in {filepath}, expected 6 columns")

    return data

def post_process_nominal_insertion_and_elevation_angles(left_right="LEFT", location="MIDDLE"):
    if left_right == "LEFT":
        vein_fp = r'Capstone/SignalProcessing/left_vein.txt'
    else:
        vein_fp = r'Capstone/SignalProcessing/right_vein.txt'
    _, _, Tz = load_data(vein_fp)

    avg_angles_per_file = []
    data_dir = 'Capstone/SignalProcessing/expert_data/right-vein/middle/'

    for filepath in glob.glob(os.path.join(data_dir, 'filtered_data_*.txt')):
        data = load_filtered_data(filepath)

        z_vals = data[:, 2]
        pitch = data[:, 3]
        roll = data[:, 4]
        yaw = data[:, 5]

        angles_to_average = []

        for i, z in enumerate(z_vals):
            if np.any(z < Tz  - 2) and np.any(z > Tz - 5):
                angles_to_average.append([pitch[i], roll[i], yaw[i]])

        if angles_to_average:
            angles_to_average = np.array(angles_to_average)
            avg_pitch = np.mean(angles_to_average[:, 0])
            avg_roll = np.mean(angles_to_average[:, 1])
            avg_yaw = np.mean(angles_to_average[:, 2])

            std_pitch = np.std(angles_to_average[:, 0])
            std_roll = np.std(angles_to_average[:, 1])
            std_yaw = np.std(angles_to_average[:, 2])

            avg_angles_per_file.append((avg_pitch, avg_roll, avg_yaw,
                                         std_pitch, std_roll, std_yaw))

    if avg_angles_per_file:
        all_avg = np.mean(avg_angles_per_file, axis=0)
        final_avg_pitch, final_avg_roll, final_avg_yaw = all_avg[:3]
        final_std_pitch, final_std_roll, final_std_yaw = all_avg[3:]

        output_path = os.path.join(data_dir, "angle_stats.txt")
        with open(output_path, "w") as f:
            f.write(f"final_avg_pitch={final_avg_pitch}, final_avg_roll={final_avg_roll}, final_avg_yaw={final_avg_yaw}, ")
            f.write(f"final_std_pitch={final_std_pitch}, final_std_roll={final_std_roll}, final_std_yaw={final_std_yaw}\n")

        return final_avg_pitch, final_avg_roll, final_avg_yaw, \
               final_std_pitch, final_std_roll, final_std_yaw
    else:
        return None, None, None, None, None, None

    

def main():
    raw = Queue()
    filtered = Queue()
    tracking_ready = Queue()

    raw_tracking = Thread(target=startTracking, args=[raw, tracking_ready], daemon=True)
    filter = Thread(target=process_file_3, args=[raw, filtered], daemon=True)

    raw_tracking.start()
    filter.start()

    raw_tracking.join()
    display('raw tracking has finished')

    filter.join()
    display('filter has finished')

    # final_avg_pitch, final_avg_roll, final_avg_yaw, final_std_pitch, final_std_roll, final_std_yaw = post_process_nominal_insertion_and_elevation_angles()

    # print(f"{final_avg_pitch=}, {final_avg_roll=}, {final_avg_yaw=}, {final_std_pitch=}, {final_std_roll=}, {final_std_yaw=}")

    

    display('Finished')

if __name__ == "__main__":
    main()