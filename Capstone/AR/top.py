import random
import threading
import multiprocessing
import logging
from threading import Thread
from queue import Queue
import sys
import os
sys.path.append(os.path.join("Capstone", "Tracking"))
from markerDetectionFrame import startTrackingPerspective
# from Tracking.markerDetectionFrame import startTrackingPerspective
from Simulation2 import mainProjection
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

    raw_tracking = Thread(target=startTrackingPerspective, args=[raw], daemon=True)
    projection = Thread(target=mainProjection, args=[raw], daemon=True)

    raw_tracking.start()

    projection.start()

    raw_tracking.join()
    display('raw tracking has finished')

    projection.join()
    display('projection has finished')


    display('Finished')

if __name__ == "__main__":
    main()