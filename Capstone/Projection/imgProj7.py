import pyautogui
import time
import os
from datetime import datetime
from AppKit import NSWorkspace
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID
)

import sys
# import os
sys.path.append(os.path.join("Capstone", "Feedback"))
from feedback3 import FeedbackUI  # Import UI


# Set up the screenshot save directory (relative path)
save_dir = "UI_Screenshots"
os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist

MAX_SCREENSHOTS = 10  # Keep only the last 10 screenshots

def cleanup_old_screenshots():
    """Deletes older screenshots, keeping only the latest MAX_SCREENSHOTS."""
    files = sorted(os.listdir(save_dir), key=lambda f: os.path.getctime(os.path.join(save_dir, f)))  # Sort by creation time
    while len(files) > MAX_SCREENSHOTS:
        os.remove(os.path.join(save_dir, files.pop(0)))  # Delete oldest file

def get_window_position(title):
    """Finds the position of a window by its title on macOS."""
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

    for window in window_list:
        window_title = window.get("kCGWindowName", "")
        if window_title and title.lower() in window_title.lower():
            bounds = window.get("kCGWindowBounds", {})
            return bounds.get("X", 0), bounds.get("Y", 0), bounds.get("Width", 0), bounds.get("Height", 0)

    return None

def capture_feedback3_window(interval=1):
    """Continuously captures screenshots of only the 'feedback3' window on macOS."""
    try:
        print(f"Looking for 'feedback3' window...")
        while True:
            position = get_window_position("Cyber-Physical Infant IV Simulator")

            if position:
                left, top, width, height = position
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                screenshot_path = os.path.join(save_dir, f"screenshot_{timestamp}.png")

                # Capture only the window region
                screenshot = pyautogui.screenshot(region=(left, top, width, height))
                screenshot.save(screenshot_path)
                print(f"Saved: {screenshot_path}")

                # Cleanup old screenshots
                cleanup_old_screenshots()
            else:
                print("Window 'feedback3' not found! Make sure it's open.")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nScreenshot capture stopped.")

# Run the function
if __name__ == "__main__":
    capture_feedback3_window(interval=1)
