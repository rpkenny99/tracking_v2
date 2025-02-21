import time
import threading
import tkinter as tk
from pynput import keyboard  # Alternative to keyboard module

# Initialize viewer position and orientation
viewer_position = [0.0, 0.0, 0.0]  # [X, Y, Z] position
viewer_orientation = [0.0, 0.0, 0.0]  # [Pitch, Yaw, Roll] angles

def update_perspective():
    """Displays the current perspective in the console and GUI."""
    position_label.config(text=f"Position: X={viewer_position[0]:.2f}, Y={viewer_position[1]:.2f}, Z={viewer_position[2]:.2f}")
    orientation_label.config(text=f"Orientation: Pitch={viewer_orientation[0]:.2f}, Yaw={viewer_orientation[1]:.2f}, Roll={viewer_orientation[2]:.2f}")

def on_press(key):
    """Handles key presses for movement and rotation."""
    global viewer_position, viewer_orientation
    try:
        if key.char == "w":  # Move forward
            viewer_position[2] += 0.05
        elif key.char == "s":  # Move backward
            viewer_position[2] -= 0.05
        elif key.char == "a":  # Move left
            viewer_position[0] -= 0.05
        elif key.char == "d":  # Move right
            viewer_position[0] += 0.05
        elif key.char == "q":  # Rotate left (yaw)
            viewer_orientation[1] -= 1.0
        elif key.char == "e":  # Rotate right (yaw)
            viewer_orientation[1] += 1.0
        elif key.char == " ":  # Move up (space bar)
            viewer_position[1] += 0.05
        elif key.char == "\t":  # Move down (tab key)
            viewer_position[1] -= 0.05
        
        update_perspective()
    except AttributeError:
        pass  # Ignore special keys

def keyboard_listener():
    """Starts a listener for keyboard input."""
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# GUI Functionality
def move_forward(): viewer_position[2] += 0.05; update_perspective()
def move_backward(): viewer_position[2] -= 0.05; update_perspective()
def move_left(): viewer_position[0] -= 0.05; update_perspective()
def move_right(): viewer_position[0] += 0.05; update_perspective()
def move_up(): viewer_position[1] += 0.05; update_perspective()
def move_down(): viewer_position[1] -= 0.05; update_perspective()
def rotate_left(): viewer_orientation[1] -= 1.0; update_perspective()
def rotate_right(): viewer_orientation[1] += 1.0; update_perspective()

# Create GUI
root = tk.Tk()
root.title("Viewer Control Console")

position_label = tk.Label(root, text="Position: X=0.00, Y=0.00, Z=0.00", font=("Arial", 14))
position_label.pack()

orientation_label = tk.Label(root, text="Orientation: Pitch=0.00, Yaw=0.00, Roll=0.00", font=("Arial", 14))
orientation_label.pack()

# Buttons for movement
tk.Button(root, text="↑ Forward", command=move_forward).pack()
tk.Button(root, text="↓ Backward", command=move_backward).pack()
tk.Button(root, text="← Left", command=move_left).pack()
tk.Button(root, text="→ Right", command=move_right).pack()
tk.Button(root, text="⬆ Up", command=move_up).pack()
tk.Button(root, text="⬇ Down", command=move_down).pack()

# Rotation buttons
tk.Button(root, text="⟲ Rotate Left", command=rotate_left).pack()
tk.Button(root, text="⟳ Rotate Right", command=rotate_right).pack()

# Run the keyboard listener in a separate thread
threading.Thread(target=keyboard_listener, daemon=True).start()

# Start the GUI loop
root.mainloop()
