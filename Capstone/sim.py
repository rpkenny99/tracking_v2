import random
import threading
import multiprocessing
import logging
from threading import Thread
from queue import Queue
from Tracking.markerDetection import startTracking
from Filter.Filter_Graphs import process_file_2
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
    max =50
    work = Queue()
    finished = Queue()

    producer = Thread(target=startTracking, args=[work], daemon=True)
    consumer = Thread(target=process_file_2, args=[work], daemon=True)

    producer.start()
    consumer.start()

    producer.join()
    display('Producer has finished')

    consumer.join()
    display('Consumer has finished')

    display('Finished')

if __name__ == "__main__":
    main()