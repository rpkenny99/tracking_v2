from SignalProcessing.signal_processing import sig_processing
from threading import Thread
from queue import Queue, Empty
from DTW.DTW_working import compute_dtw
import os
import time
import numpy as np

def remove_first_n_lines(file_path, file_lock, n=10):
    with file_lock:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if len(lines) > n:
            with open(file_path, 'w') as file:
                file.writelines(lines[n:])

def load_angle_stats(filepath):
    with open(filepath, 'r') as f:
        line = f.readline().strip()

    # Convert line to dict
    parts = line.split(", ")
    stats = {}
    for part in parts:
        key, val = part.split("=")
        stats[key] = float(val)

    return (
        stats["final_avg_pitch"],
        stats["final_avg_roll"],
        stats["final_avg_yaw"],
        stats["final_std_pitch"],
        stats["final_std_roll"],
        stats["final_std_yaw"],
    )

def get_expert_data_file_path(vein, location):
    if vein == "Left Vein":
        if location == "Point B":
            fp = r"Capstone/SignalProcessing/expert_data/left-vein/middle"
        if location == "Point A":
            fp = r"Capstone/SignalProcessing/expert_data/left-vein/bottom"
        if location == "Point C":
            fp = r"Capstone/SignalProcessing/expert_data/left-vein/top"
    if vein == "Right Vein":
        if location == "Point B":
            fp = r"Capstone/SignalProcessing/expert_data/right-vein/middle"
        if location == "Point A":
            fp = r"Capstone/SignalProcessing/expert_data/right-vein/bottom"
        if location == "Point C":
            fp = r"Capstone/SignalProcessing/expert_data/right-vein/top"
    return fp

def get_average_insertion_and_elevation_angles(fp):
    fp = os.path.join(fp, "angle_stats.txt")
    return load_angle_stats(fp)

def monitor(filtered,
            sig_processed,
            app_to_signal_processing,
            angle_range_queue,
            simulation_running_queue,
            lock,
            focal_point_queue,
            direction_intruction_queue,
            user_score_queue):
    """
    Receives live trajectory data, finds the closest mean trajectory point, and 
    checks if it's within the standard deviation bounds.
    """
    print("Feedback Monitor started")
    signal_processor = None
    control = Queue()
    fp = None
    while True:
        print("Waiting on user input...")
        vein, location = app_to_signal_processing.get(block=True)
        # Case where feedback is running and we get a new item in the queue. It will always be to end sim
        if vein is None and location is None:
            control.put(0)
            print("Received stop signal")
            signal_processor.join()
            signal_processor = None
            print("Ending Signal Processing...\n")
            simulation_running_queue.put(0)

            time.sleep(2)

            remove_first_n_lines(r"Capstone/Filter/filtered_data.txt", lock, 10)
         
            # Call Dynamic Time Warping
            average_similarity = compute_dtw(file_lock=lock)
            print(f"{average_similarity=}")

            if not average_similarity:
                user_score_percentage = -1
            else:
                k = 0.00125  # Decay rate
                user_score_percentage = round(100 * 10**(-k * average_similarity), 1)
                print(f"{np.exp(-k * average_similarity)=}, {100 * 10**(-k * average_similarity)}")


            print(f"Got score {user_score_percentage=}\n")

            user_score_queue.put(user_score_percentage)

            print('Put user score in queue')

            # Clear this queue
            while True:
                try:
                    direction_intruction_queue.get_nowait()  # Non-blocking
                except Empty:
                    break  # Stop when queue is empty

        else:
            # If any of the setup conditions are none, keep polling for the rest.
            print(f"{vein=}, {location=}\n")

            focal_point_queue.put(vein)

            fp = get_expert_data_file_path(vein, location)
            expert_pitch, expert_roll, expert_yaw, expert_pitch_std, expert_roll_std, expert_yaw_std = get_average_insertion_and_elevation_angles(fp)
            angle_range_queue.put([expert_pitch, expert_roll, expert_yaw, expert_pitch_std, expert_roll_std, expert_yaw_std])

            signal_processor = Thread(target=sig_processing, args=[filtered, sig_processed, control, direction_intruction_queue], daemon=True)
            signal_processor.start()

            simulation_running_queue.put(1)

            print(f"{expert_pitch=}")

            

