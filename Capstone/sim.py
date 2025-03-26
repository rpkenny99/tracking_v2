import random
import threading
import multiprocessing
import logging
from threading import Thread
from queue import Queue
from Tracking.markerDetection import startTracking
from Filter.Filter_Graphs import process_file_2
from SignalProcessing.signal_processing import sig_processing
from SignalProcessing.feedback_monitor import monitor
from Feedback.updated_feeback import MainApplication
import time
logging.basicConfig(format='%(levelname)s - %(asctime)s.%(msecs)03d: %(message)s',datefmt='%H:%M:%S', level=logging.DEBUG)

from multiprocessing import Lock

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

def main():
    raw = Queue()
    filtered = Queue()
    tracking_ready = Queue()
    sig_processed = Queue()
    app_to_signal_processing = Queue() # Sim_run, vein, location
    angle_range_queue = Queue()
    simulation_running_queue = Queue()
    focal_point_queue = Queue()
    direction_intruction_queue = Queue()

    file_lock = Lock()

    raw_tracking = Thread(target=startTracking, args=[raw, tracking_ready, focal_point_queue], daemon=True)
    filter = Thread(target=process_file_2, args=[raw, filtered, simulation_running_queue, file_lock], daemon=True)
    feedback_monitor = Thread(target=monitor, args=[filtered,
                                                    sig_processed,
                                                    app_to_signal_processing,
                                                    angle_range_queue,
                                                    simulation_running_queue,
                                                    file_lock,
                                                    focal_point_queue,
                                                    direction_intruction_queue], daemon=True)
    # signal_processing = Thread(target=sig_processing, args=[filtered, sig_processed, app_to_signal_processing, angle_range_queue], daemon=True)

    raw_tracking.start()
    filter.start()
    feedback_monitor.start()

    # Run the PyQt GUI in the main thread
    
    main_app = MainApplication(sig_processed,
                               app_to_signal_processing,
                               angle_range_queue,
                               direction_intruction_queue)
    while True:
        if tracking_ready.get() == 1:
            break

    main_app.run()  # This blocks execution``

    raw_tracking.join()
    display('raw tracking has finished')

    filter.join()
    display('filter has finished')

    feedback_monitor.join()
    display('signal processing has finished')

    display('Finished')

if __name__ == "__main__":
    main()