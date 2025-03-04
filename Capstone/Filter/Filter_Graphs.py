import os
import time
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_FILTERED_DATA_FILE_PATH = "Capstone/Filter/filtered_data.txt"


def scipy_low(cutoff_freq, sample_time, x0, x1, x2, y1, y2):
    """
    Perform low-pass filtering using a 2nd-order Butterworth filter.
    """
    nyquist = 1 / (2 * sample_time)
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    y = (b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2) / a[0]
    return y

def process_file(file_path, output_file, cutoff_freq=5, sample_time=0.02, monitor_mode=True, wait_time=5):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    buffers = {key: [0, 0, 0] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}
    outputs = {key: [0, 0] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}

    raw_data = {key: [] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}  # Store raw data
    filtered_data = {key: [] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}  # Store filtered data

    last_position = 0
    start_wait_time = None

    try:
        with open(file_path, 'r') as file, open(output_file, 'w') as output:
            while True:
                file.seek(last_position)
                lines = file.readlines()
                last_position = file.tell()

                if lines:  # New data found
                    start_wait_time = None  # Reset wait timer
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            # Parse data
                            x0, y0, z0, roll0, pitch0, yaw0 = map(float, line.split())

                            # Store raw data
                            for key, value in zip(["x", "y", "z", "roll", "pitch", "yaw"],
                                                  [x0, y0, z0, roll0, pitch0, yaw0]):
                                raw_data[key].append(value)

                            # Update buffers and compute filtered values
                            for key, value in zip(
                                    ["x", "y", "z", "roll", "pitch", "yaw"],
                                    [x0, y0, z0, roll0, pitch0, yaw0],
                            ):
                                buffers[key] = [value] + buffers[key][:2]
                                outputs[key] = [
                                                   scipy_low(
                                                       cutoff_freq, sample_time,
                                                       buffers[key][0], buffers[key][1], buffers[key][2],
                                                       outputs[key][0], outputs[key][1],
                                                   )
                                               ] + outputs[key][:1]

                                # Store filtered data
                                filtered_data[key].append(outputs[key][0])

                            # Write filtered data to the output file immediately
                            filtered_row = [
                                outputs["x"][0], outputs["y"][0], outputs["z"][0],
                                outputs["roll"][0], outputs["pitch"][0], outputs["yaw"][0],
                            ]
                            output.write(" ".join(map(str, filtered_row)) + "\n")
                            output.flush()  # Ensure real-time writing to the file

                        except ValueError as e:
                            print(f"Error processing line: {line}\n{e}")

                elif start_wait_time is None:
                    start_wait_time = time.time()

                elif time.time() - start_wait_time > wait_time:
                    print(f"No new data detected in the last {wait_time} seconds. Stopping monitoring.")
                    break

                if not monitor_mode:
                    break

                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")

def remove_first_n_lines(file_path, n=6):
    """
    Remove the first `n` lines from a file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    if len(lines) > n:
        with open(file_path, 'w') as file:
            file.writelines(lines[n:])

buffers = {key: [0, 0, 0] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}
outputs = {key: [0, 0] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}

raw_data = {key: [] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}  # Store raw data
filtered_data = {key: [] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}  # Store filtered data

def process_file_2(raw_data_queue, 
                   output_file=DEFAULT_FILTERED_DATA_FILE_PATH,
                   cutoff_freq=5,
                   sample_time=0.02,
                   monitor_mode=True,
                   wait_time=5):
    global outputs
    global buffers
    global raw_data

    start_wait_time = None

    try:
        with open(output_file, 'w') as output:
            while True:
                raw_data_entry = raw_data_queue.get()               # blocking get
                if raw_data_entry is None:               # <-- sentinel
                    break
                # Otherwise, process item
                print(f"Consumed: {raw_data_entry}")

                                    # Parse data
                x0, y0, z0, roll0, pitch0, yaw0 = raw_data_entry

                # Update buffers and compute filtered values
                for key, value in zip(
                        ["x", "y", "z", "roll", "pitch", "yaw"],
                        [x0, y0, z0, roll0, pitch0, yaw0],
                ):
                    raw_data[key].append(value)
                    buffers[key] = [value] + buffers[key][:2]
                    outputs[key] = [
                                        scipy_low(
                                            cutoff_freq, sample_time,
                                            buffers[key][0], buffers[key][1], buffers[key][2],
                                            outputs[key][0], outputs[key][1],
                                        )
                                    ] + outputs[key][:1]

                    # Store filtered data
                    filtered_data[key].append(outputs[key][0])

                    # Write filtered data to the output file immediately
                    filtered_row = [
                        outputs["x"][0], outputs["y"][0], outputs["z"][0],
                        outputs["roll"][0], outputs["pitch"][0], outputs["yaw"][0],
                    ]
                output.write(" ".join(map(str, filtered_row)) + "\n")
                output.flush()  # Ensure real-time writing to the file

        # plot_data(raw_data, filtered_data)
        # plot_filtered_and_translated_data("Capstone/Filter/filtered_data.txt", "left_vein_final.txt", "right_vein_final.txt")
        remove_first_n_lines(output_file, n=10)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")


def plot_data(raw_data, filtered_data):
    """
    Generate graphs to compare raw and filtered data for all six parameters.
    """
    parameters = ["x", "y", "z", "roll", "pitch", "yaw"]
    plt.figure(figsize=(12, 10))

    for i, param in enumerate(parameters):
        plt.subplot(3, 2, i + 1)
        plt.plot(raw_data[param], label='Raw', linestyle='--', alpha=0.7)
        plt.plot(filtered_data[param], label='Filtered', linewidth=2)
        plt.title(f'{param.capitalize()} Data')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.5)

    plt.show()

def plot_filtered_and_translated_data(filtered_file, left_vein_file, right_vein_file):
    """
    Plot the filtered x, y, z trajectory data along with the translated left and right veins.
    """
    # Load filtered data
    filtered_data = np.loadtxt(filtered_file)
    
    # Load translated vein data
    left_vein_translated = np.loadtxt(left_vein_file)
    right_vein_translated = np.loadtxt(right_vein_file)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot filtered data in red
    ax.plot(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2], 'r-', label="Filtered Trajectory")
    
    # Plot translated veins in green
    ax.plot(left_vein_translated[:, 0], left_vein_translated[:, 1], left_vein_translated[:, 2], 'g-', label="Left Vein Translated")
    ax.plot(right_vein_translated[:, 0], right_vein_translated[:, 1], right_vein_translated[:, 2], 'g-', label="Right Vein Translated")
    
    # Labels and title
    ax.set_xlabel("Tx")
    ax.set_ylabel("Ty")
    ax.set_zlabel("Tz")
    ax.set_title("Filtered Data and Translated Veins")
    ax.legend()
    
    plt.show()

# Example usage:
# 

# def main():
#     print("Real-Time Data Filtering Program")
#     base_directory = input("Enter the directory containing the data file: ").strip()
#     input_file_name = input("Enter the name of the input data file: ").strip()
#     output_file_name = input("Enter the name of the output file: ").strip()
#     file_path = os.path.join(base_directory, input_file_name)
#     output_path = os.path.join(base_directory, output_file_name)

#     if not os.path.exists(file_path):
#         print(f"Error: The file '{file_path}' does not exist.")
#         return

#     # Ask user if they want to monitor for new data
#     mode = input("Do you want to monitor for new data? (yes/no): ").strip().lower()
#     monitor_mode = mode == 'yes'

#     # Start filtering
#     try:
#         process_file(file_path, output_path, monitor_mode=monitor_mode)
#     except Exception as e:
#         print(f"An error occurred: {e}")



if __name__ == "__main__":
    plot_filtered_and_translated_data("Capstone/Filter/filtered_data.txt", "left_vein_final.txt", "right_vein_final.txt")