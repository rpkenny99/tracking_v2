import numpy as np
import time
from functools import wraps
from SignalProcessing.compute_avg_std_dev import get_mean_std_bounds

def time_it(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        # print(f"{func.__name__} executed in {end_time - start_time:.6f} seconds")
        return result
    return wrapper

@time_it  # Wraps function to measure execution time
def find_nearest_mean_index(live_data, mean_traj, prev_idx, search_radius=100):
    """
    Finds the closest point in the mean trajectory to the live data.
    Uses a search window of ± search_radius around the previous index to optimize performance.

    Args:
    - live_data: np.array of shape (6,), the live trajectory point.
    - mean_traj: np.array of shape (N, 6), the precomputed mean trajectory.
    - prev_idx: int, index of the previous closest point (None if first call).
    - search_radius: int, number of indices to search around the previous closest index.

    Returns:
    - closest_idx: int, index of the closest trajectory point.
    """
    if prev_idx is None:
        # First call: Search the entire trajectory
        search_range = np.arange(len(mean_traj))
    else:
        # Limit the search range to ± search_radius around the last index
        start = max(prev_idx - search_radius, 0)
        end = min(prev_idx + search_radius + 1, len(mean_traj))
        search_range = np.arange(start, end)

    # Compute distances only for the selected range
    distances = np.linalg.norm(mean_traj[search_range, :3] - live_data[:3], axis=1)
    
    # Find the index of the minimum distance in the local window
    min_idx_local = np.argmin(distances)
    
    # Convert local index to global index
    closest_idx = search_range[min_idx_local]

    return closest_idx

# Define separate scaling factors for Pro vs. Amateur
STD_PRO = 1.8
STD_AMATEUR = 2.5

def is_within_bounds(live_data, mean_traj, upper_bound, lower_bound, window_size=10000):
    """
    Checks if the live trajectory pose (x, y, z, pitch, roll, yaw) is within bounds.
    Differentiates between "Pro" (stricter) and "Amateur" (looser) tolerance ranges.
    - Finds the nearest mean trajectory point.
    - Expands the check to include ±window_size indices around the nearest mean point.
    - Uses different multipliers for Pro vs Amateur bounds.
    """
    global prev_idx
    curr_idx = find_nearest_mean_index(live_data, mean_traj, prev_idx=prev_idx)
    # print(f"Initial closest index: {curr_idx}")

    # Define search window around the closest index
    start_idx = max(curr_idx - window_size, 0)
    end_idx = min(curr_idx + window_size + 1, len(mean_traj))

    # Extract relevant section of the bounds
    mean_window = mean_traj[start_idx:end_idx]

    live_data = np.array(live_data)

    # Slice only the first 3 entries
    live_data = live_data[:3]  # Ensuring it's (N, 3)

    # Ensure upper_bound and lower_bound are NumPy arrays
    upper_bound = np.array(upper_bound)
    lower_bound = np.array(lower_bound)

    # Ensure upper_bound and lower_bound are properly shaped
    upper_bound = upper_bound[start_idx:end_idx, :3]  # Extract first three entries
    lower_bound = lower_bound[start_idx:end_idx, :3]  # Extract first three entries

    # Scale the standard deviation bounds separately for Pro & Amateur (only for first three entries)
    upper_pro = mean_window[:, :3] + (STD_PRO * (upper_bound - mean_window[:, :3]))
    lower_pro = mean_window[:, :3] - (STD_PRO * (mean_window[:, :3] - lower_bound))

    upper_amateur = mean_window[:, :3] + (STD_AMATEUR * (upper_bound - mean_window[:, :3]))
    lower_amateur = mean_window[:, :3] - (STD_AMATEUR * (mean_window[:, :3] - lower_bound))

    # Check if live data is within the Pro range
    within_upper_pro = False
    within_lower_pro = False
    for i, entry in enumerate(upper_pro):
        if np.all(live_data < entry):
            within_upper_pro = True
            if np.all(live_data > lower_pro[i]):
                within_lower_pro = True
                break
            else:
                within_lower_pro = False
        else:
            within_upper_pro = False
    within_pro = (within_upper_pro & within_lower_pro)

    # Check if live data is within the Pro range
    within_upper_amateur = False
    within_lower_amateur = False
    for i, entry in enumerate(upper_amateur):
        if np.all(live_data < entry):
            within_upper_amateur = True
            if np.all(live_data > lower_amateur[i]):
                within_lower_amateur = True
                break
            else:
                within_lower_amateur = False
        else:
            within_upper_amateur = False
    within_amateur = (within_upper_amateur & within_lower_amateur)

    prev_idx = curr_idx

    return within_pro, within_amateur, curr_idx

def load_data(filename):
    """Loads Tx, Ty, Tz values from a text file."""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1], data[:, 2]

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

def get_average_insertion_and_elevation_angles(vein, location):
    if vein == "Left Vein":
        if location == "Point B":
            fp = r"Capstone/SignalProcessing/expert_data/left-vein/middle/angle_stats.txt"
            return load_angle_stats(fp)

def sig_processing(filtered_data_queue, sig_processed_queue, app_to_signal_processing, angle_range_queue):
    """
    Receives live trajectory data, finds the closest mean trajectory point, and 
    checks if it's within the standard deviation bounds.
    """
    global prev_idx
    prev_idx = None
    (mean_traj, upper_bound, lower_bound), _ = get_mean_std_bounds()

    expert_pitch, expert_roll, expert_yaw, expert_pitch_std, expert_roll_std, expert_yaw_std = None, None, None, None, None, None

    feedback_started = False

    _, _, Tz_left = load_data(r'Capstone/SignalProcessing/left_vein.txt')
    _, _, Tz_right = load_data(r'Capstone/SignalProcessing/right_vein.txt')

    while True:
        # Case where feedback is running and we get a new item in the queue. It will always be to end sim
        if not app_to_signal_processing.empty() and feedback_started:
            app_to_signal_processing.get()
            feedback_started = False

        # Case where feedback is not running and queue is empty. No action required to continue.
        elif app_to_signal_processing.empty() and not feedback_started:
            continue

        # Case where the queue is not empty and feedback has not started. Meaning there is some setup information
        elif not app_to_signal_processing.empty():
            sim_running, vein, location = app_to_signal_processing.get()
            # If any of the setup conditions are none, keep polling for the rest.
            print(f"{sim_running}, {vein=}, {location=}")
            if sim_running is None or vein is None or location is None:
                continue
            else:
                feedback_started = True
                expert_pitch, expert_roll, expert_yaw, expert_pitch_std, expert_roll_std, expert_yaw_std = get_average_insertion_and_elevation_angles(vein, location)
                angle_range_queue.put([expert_pitch, expert_roll, expert_yaw, expert_pitch_std, expert_roll_std, expert_yaw_std])

                print(f"{expert_pitch=}")

        
        filtered_data_entry = filtered_data_queue.get()
        if feedback_started:
            if filtered_data_entry is None:
                print("Signal Processor Dying...")
                sig_processed_queue.put(None)
                break

            # print(f"Received Live Data: {filtered_data_entry}")
            if sig_processed_queue.empty():
                sig_processed_queue.put(filtered_data_entry)

            if np.any(filtered_data_entry[2] < (Tz_left + 15)) or np.any(filtered_data_entry[2] < (Tz_right + 15)): 
                # Check if the live data is within bounds
                within_bounds_pro, within_bounds_amateur, nearest_idx = is_within_bounds(filtered_data_entry, mean_traj, upper_bound, lower_bound)

                if within_bounds_pro:
                    print(f"✅")
                elif within_bounds_amateur:
                    print(f"🟨")
                else:
                    print(f"❌")
