import numpy as np
import time
from functools import wraps
from SignalProcessing.compute_avg_std_dev import get_mean_std_bounds
import queue

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
STD_AMATEUR = 4

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

def get_direction_to_correct_trajectory(live_data, mean_traj, nearest_idx):
    """
    Determines the direction the needle should move to approach the mean trajectory.

    Args:
    - live_data: np.array of shape (6,) representing (x, y, z, pitch, roll, yaw)
    - mean_traj: np.array of shape (N, 6)
    - nearest_idx: int, index of closest mean trajectory point

    Returns:
    - direction: str, human-readable direction suggestion
    """
    directions = []

    # Extract 3D positions
    live_xyz = np.array(live_data[:3])
    mean_xyz = np.array(mean_traj[nearest_idx][:3])
    delta = mean_xyz - live_xyz

    threshold = 2.5

    if delta[0] > threshold:
        directions.append("right")
    elif delta[0] < -threshold:
        directions.append("left")

    if delta[2] > threshold:
        directions.append("up")
    elif delta[2] < -threshold:
        directions.append("down")

    if not directions:
        return None
    
    return directions

def sig_processing(filtered_data_queue, sig_processed_queue, control, direction_intruction_queue):
    """
    Receives live trajectory data, finds the closest mean trajectory point, and 
    checks if it's within the standard deviation bounds.
    """
    global prev_idx
    prev_idx = None
    (mean_traj, upper_bound, lower_bound), _ = get_mean_std_bounds()

    while True:
        if not control.empty():
            control.get()
            break
        try:
            # Wait up to 1 second for new data
            filtered_data_entry = filtered_data_queue.get(timeout=0.5)
            # Process data
        except queue.Empty:
            # If no data arrived within 1s, loop again, check stop_event
            continue

        if filtered_data_entry is None:
            print("Signal Processor Dying...")
            sig_processed_queue.put(None)
            break

        # print(f"Received Live Data: {filtered_data_entry}")
        if sig_processed_queue.empty():
            sig_processed_queue.put(filtered_data_entry)

        # Check if the live data is within bounds
        within_bounds_pro, within_bounds_amateur, nearest_idx = is_within_bounds(filtered_data_entry, mean_traj, upper_bound, lower_bound)

        if within_bounds_pro:
            print(f"✅")
            pass
        elif within_bounds_amateur:
            print(f"🟨")
            pass
        else:
            dir = get_direction_to_correct_trajectory(filtered_data_entry, mean_traj, nearest_idx)
            direction_intruction_queue.put(dir)
            # print(f"{dir=}")
            # print(f"❌")
