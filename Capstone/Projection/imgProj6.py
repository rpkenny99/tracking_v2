import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt6.QtGui import QPixmap, QImage, QGuiApplication
from PyQt6.QtCore import Qt
import sys
import os
sys.path.append(os.path.join("Capstone", "Feedback"))
from feedback3 import FeedbackUI  # Import UI

class UI_Display(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UI Display")
        self.setGeometry(100, 100, 600, 400)
        
        self.label = QLabel("Overlay UI", self) #creates text "Overlay UI" on the window
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter) #Centres the text
        
        #Organizing widgets
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def capture_ui(self):
        """Capture the UI display and return as an OpenCV image."""
        pixmap = QGuiApplication.primaryScreen().grabWindow(0)  # Grab window contents
        image = pixmap.toImage()
        width, height = image.width(), image.height()
        
        # Convert QImage to OpenCV format
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # RGBA format
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)  # Convert to BGR

    def apply_transformation(self, img):
        """Apply perspective transformation for a 25-degree projection plane."""
        h, w = img.shape[:2]
        
        # Define source points (original image corners)
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Define destination points (simulated 25-degree tilt projection)
        offset = int(w * 0.466)  # Approximate vertical shift due to tilt
        dst_pts = np.float32([[0, offset], [w, 0], [w, h], [0, h - offset]])
        
        # Compute perspective transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        transformed_img = cv2.warpPerspective(img, matrix, (w, h))
        return transformed_img
    
    def show_transformed_ui(self):
        """Display the transformed UI overlay."""
        img = self.capture_ui()
        transformed_img = self.apply_transformation(img)
        
        # Display using OpenCV
        cv2.imshow("Transformed UI", transformed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = UI_Display()
    ui.show()
    app.exec()
    
    # Capture and transform UI after closing window
    ui.show_transformed_ui()
