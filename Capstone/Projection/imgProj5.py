import sys
import Quartz
import objc
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer
from PIL import Image
import numpy as np

class ScreenCaptureApp(QMainWindow):
    def __init__(self, target_app_name="MyApp"):
        super().__init__()

        self.target_app_name = target_app_name
        self.target_window_id = self.get_window_id(self.target_app_name)

        self.setWindowTitle("Live Application Capture")
        self.setGeometry(100, 100, 800, 500)

        # QLabel to display the captured snapshot
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Layout setup
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_snapshot)
        self.timer.start(100)  # Update every 100ms

    def get_window_id(self, app_name):
        """Get the window ID of the target application."""
        window_list = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
        for window in window_list:
            if window.get('kCGWindowOwnerName', '') == app_name:
                return window['kCGWindowNumber']
        return None

    def capture_window(self, window_id):
        """Capture a screenshot of the window with the given ID."""
        image_ref = Quartz.CGWindowListCreateImage(
            Quartz.CGRectInfinite,
            Quartz.kCGWindowListOptionIncludingWindow,
            window_id,
            Quartz.kCGWindowImageDefault
        )
        if not image_ref:
            return None

        width = Quartz.CGImageGetWidth(image_ref)
        height = Quartz.CGImageGetHeight(image_ref)
        bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)
        data_provider = Quartz.CGImageGetDataProvider(image_ref)
        data = Quartz.CGDataProviderCopyData(data_provider)

        img = Image.frombytes("RGBA", (width, height), data, "raw", "RGBA", bytes_per_row)
        return img

    def update_snapshot(self):
        """Capture and update the snapshot of the target application in real-time."""
        if self.target_window_id:
            screenshot = self.capture_window(self.target_window_id)
            if screenshot:
                img = screenshot.convert("RGB")  # Convert to RGB format
                data = np.array(img)  # Convert PIL image to numpy array
                height, width, channel = data.shape
                bytes_per_line = 3 * width
                qimage = QImage(data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            else:
                self.label.setText(f"Window '{self.target_app_name}' not found")
        else:
            self.label.setText(f"Window '{self.target_app_name}' not found")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Replace "MyApp" with the exact window title of your application
    window = ScreenCaptureApp(target_app_name="MyApp")

    window.show()
    sys.exit(app.exec())
