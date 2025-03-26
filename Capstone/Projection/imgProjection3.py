import sys
import time
import random
from queue import Queue
from PyQt6.QtWidgets import QApplication
from updated_feeback import FeedbackUI

def generate_fake_data(work_queue):
    """Simulates real-time sensor input by pushing random values into the queue."""
    while True:
        x = round(random.uniform(-5, 5), 2)
        y = round(random.uniform(-5, 5), 2)
        z = round(random.uniform(-5, 5), 2)
        pitch = round(random.uniform(-30, 30), 2)  # Simulated angle
        roll = round(random.uniform(-30, 30), 2)
        yaw = round(random.uniform(-30, 30), 2)  # Depth deviation

        work_queue.put([x, y, z, pitch, roll, yaw])

        time.sleep(0.1)  # Simulating 10Hz data stream (100ms delay per update)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Queues for real-time data communication
    work_queue = Queue()
    angle_range_queue = Queue()
    app_to_signal_processing = Queue()

    # **Initialize the Feedback UI**
    selected_vein = "Left Vein"
    selected_point = "Point B"

    feedback_ui = FeedbackUI(
        selected_vein,
        selected_point,
        work_queue=work_queue,
        angle_range_queue=angle_range_queue,
        app_to_signal_processing=app_to_signal_processing
    )

    feedback_ui.show()

    # Start real-time data simulation (Fake sensor data generator)
    import threading
    data_thread = threading.Thread(target=generate_fake_data, args=(work_queue,), daemon=True)
    data_thread.start()

    sys.exit(app.exec())
