import random
import threading
import multiprocessing
import logging
from threading import Thread
from queue import Queue
from Tracking.markerDetection import startTracking
from Filter.Filter_Graphs import process_file_3
from SignalProcessing.signal_processing import sig_processing
import time
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

def main():
    raw = Queue()
    filtered = Queue()

    raw_tracking = Thread(target=startTracking, args=[raw], daemon=True)
    filter = Thread(target=process_file_3, args=[raw, filtered], daemon=True)

    raw_tracking.start()
    filter.start()

    raw_tracking.join()
    display('raw tracking has finished')

    filter.join()
    display('filter has finished')

    display('Finished')

if __name__ == "__main__":
    main()