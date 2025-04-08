import sys
import threading
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit
from PyQt6.QtCore import QThread, pyqtSignal
import time

class WorkerThread(QThread):
    update_signal = pyqtSignal(str)

    def run(self):
        for i in range(5):
            time.sleep(2)
            self.update_signal.emit(f"Processing step {i+1}/5 completed...")
        self.update_signal.emit("Processing finished.")

class MainApplication(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Interactive GUI")
        self.setGeometry(100, 100, 400, 300)
        
        self.layout = QVBoxLayout()
        
        self.label = QLabel("Welcome to the Interactive GUI")
        self.layout.addWidget(self.label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.layout.addWidget(self.log_output)
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)
        
        self.setLayout(self.layout)

    def start_processing(self):
        self.log_output.append("Starting background processing...")
        self.worker = WorkerThread()
        self.worker.update_signal.connect(self.update_log)
        self.worker.start()

    def update_log(self, message):
        self.log_output.append(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApplication()
    main_app.show()
    sys.exit(app.exec())
