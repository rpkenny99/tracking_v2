import os
import time
from scipy.signal import butter, lfilter

def scipy_low(cutoff_freq, sample_time, x0, x1, x2, y1, y2):
    """
    Perform low-pass filtering using a 2nd-order Butterworth filter.
    """
    nyquist = 1 / (2 * sample_time)
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    y = (b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2) / a[0]
    return y


def process_file(file_path, output_file, cutoff_freq=5, sample_time=0.02, monitor_mode=True, timeout=10):
    """
    Process data from a file, either in static mode (process all data at once) or dynamic mode (monitor for new data).

    :param file_path: Path to the input data file.
    :param output_file: Path to the output filtered data file.
    :param cutoff_freq: Cutoff frequency for the low-pass filter.
    :param sample_time: Sampling time for the data.
    :param monitor_mode: If True, monitor for new data. If False, process file once and exit.
    :param timeout: Timeout in seconds to stop monitoring if no new data is found.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    buffers = {key: [0, 0, 0] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}
    outputs = {key: [0, 0] for key in ["x", "y", "z", "roll", "pitch", "yaw"]}

    last_position = 0
    last_data_time = time.time()

    try:
        with open(file_path, 'r') as file, open(output_file, 'w') as output:
            while True:
                file.seek(last_position)
                lines = file.readlines()
                last_position = file.tell()

                if lines:
                    last_data_time = time.time()  # Reset timeout timer
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            # Parse data
                            x0, y0, z0, roll0, pitch0, yaw0 = map(float, line.split())

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

                            # Write filtered data to the output file
                            filtered_data = [
                                outputs["x"][0], outputs["y"][0], outputs["z"][0],
                                outputs["roll"][0], outputs["pitch"][0], outputs["yaw"][0],
                            ]
                            output.write(" ".join(map(str, filtered_data)) + "\n")

                        except ValueError as e:
                            print(f"Error processing line: {line}\n{e}")

                    print("Filtered data written to output file.")

                if not monitor_mode:
                    break  # Exit after processing the file in static mode

                if time.time() - last_data_time > timeout:
                    print("No new data detected. Stopping monitoring.")
                    break

                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")


# Main Program
def main():
    print("Real-Time Data Filtering Program")
    base_directory = input("Enter the directory containing the data file: ").strip()
    input_file_name = input("Enter the name of the input data file: ").strip()
    output_file_name = input("Enter the name of the output file: ").strip()
    file_path = os.path.join(base_directory, input_file_name)
    output_path = os.path.join(base_directory, output_file_name)

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Ask user if they want to monitor for new data
    mode = input("Do you want to monitor for new data? (yes/no): ").strip().lower()
    monitor_mode = mode == 'yes'

    # Start filtering
    try:
        process_file(file_path, output_path, monitor_mode=monitor_mode)
    except Exception as e:
        print(f"An error occurred: {e}")

    # try:
    #     base_directory = input("Enter the directory containing the data file: ").strip()
    #     input_file_name = input("Enter the name of the input data file: ").strip()
    #     output_file_name = input("Enter the name of the output file: ").strip()
    #     file_path = os.path.join(base_directory, input_file_name)
    #     output_path = os.path.join(base_directory, output_file_name)
    #
    #     if not os.path.exists(file_path):
    #         print(f"Error: The file '{file_path}' does not exist.")
    #         return
    #
    #     # Ask user if they want to monitor for new data
    #     mode = input("Do you want to monitor for new data? (yes/no): ").strip().lower()
    #     monitor_mode = mode == 'yes'
    #
    #     # Start filtering
    #     process_file(file_path, output_path, monitor_mode=monitor_mode)
    # except KeyboardInterrupt:
    #     print("\nProgram interrupted by user. Exiting...")


if __name__ == "__main__":
    main()
