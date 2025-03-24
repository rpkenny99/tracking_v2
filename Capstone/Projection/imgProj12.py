import sys
import math
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsProxyWidget
from PyQt6.QtGui import QTransform
from PyQt6.QtCore import QPointF
from feedback3 import FeedbackUI  # Ensure feedback3.py is in your PYTHONPATH

import cv2
import numpy as np
from queue import Queue

def main():
    app = QApplication(sys.argv)

    # Instantiate the interactive UI with the required parameters.
    selected_vein = "Left Vein"
    selected_point = "Point A"
    feedback_ui = FeedbackUI(selected_vein, selected_point)
    
    # Create a QGraphicsScene and embed the UI via a QGraphicsProxyWidget.
    scene = QGraphicsScene()
    proxy = QGraphicsProxyWidget()
    proxy.setWidget(feedback_ui)
    scene.addItem(proxy)

    # Create a QGraphicsView to display the scene.
    view = QGraphicsView(scene)
    view.setWindowTitle("Interactive Transformed UI")

    # Ensure the widget is laid out so that we can obtain its dimensions.
    feedback_ui.show()
    app.processEvents()  # Process events to update layout

    # Obtain the original widget's geometry.
    rect = feedback_ui.rect()
    W = rect.width()
    H = rect.height()

    # Define source points (corners of the original widget)
    src_points = [
        QPointF(0, 0),    # top-left
        QPointF(W, 0),    # top-right
        QPointF(W, H),    # bottom-right
        QPointF(0, H)     # bottom-left
    ]

    # Compute destination points.
    # The top edge is "compressed" to simulate a 25° tilt.
    cos25 = math.cos(math.radians(25))
    new_top_width = W * cos25
    dx = (W - new_top_width) / 2  # Offset to center the compressed top edge

    dst_points = [
        QPointF(dx, 0),         # new top-left
        QPointF(W - dx, 0),       # new top-right
        QPointF(W, H),            # bottom-right remains the same
        QPointF(0, H)             # bottom-left remains the same
    ]

    # Compute the transformation using QTransform.quadToQuad.
    transform = QTransform()
    success = QTransform.quadToQuad(src_points, dst_points, transform)
    if not success:
        print("quadToQuad transformation failed")
    else:
        proxy.setTransform(transform)

    # Show the view with the transformed, yet interactive, UI.
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()