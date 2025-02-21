import keyboard
import time
import threading
import tkinter as tk

# Initialize viewer position and orientation
viewer_position = [0.0, 0.0, 0.0]  # [X, Y, Z] position
viewer_orientation = [0.0, 0.0, 0.0]  # [Pitch, Yaw, Roll] angles

def update_perspective():
    """Displays the current perspective in the console and GUI."""
    position_label.config(text=f"Position: X={viewer_position[0]:.2f}, Y={viewer_position[1]:.2f}, Z={viewer_position[2]:.2f}")
    orientation_label.config(text=f"Orientation: Pitch={viewer_orientation[0]:.2f}, Yaw={viewer_orientation[1]:.2f}, Roll={viewer_orientation[2]:.2f}")

def manual_control():
    """Simulates head movement using keyboard controls."""
    global viewer_position, viewer_orientation

    print("Use W/A/S/D to move, Q/E to rotate, and ESC to quit.")
    
    while True:
        # Movement controls
        if keyboard.is_pressed("w"):  # Move forward
            viewer_position[2] += 0.05  
        if keyboard.is_pressed("s"):  # Move backward
            viewer_position[2] -= 0.05  
        if keyboard.is_pressed("a"):  # Move left
            viewer_position[0] -= 0.05  
        if keyboard.is_pressed("d"):  # Move right
            viewer_position[0] += 0.05  
        if keyboard.is_pressed("space"):  # Move up
            viewer_position[1] += 0.05  
        if keyboard.is_pressed("shift"):  # Move down
            viewer_position[1] -= 0.05  

        # Rotation controls
        if keyboard.is_pressed("q"):  # Rotate left (yaw)
            viewer_orientation[1] -= 1.0  
        if keyboard.is_pressed("e"):  # Rotate right (yaw)
            viewer_orientation[1] += 1.0  

        # Update and display perspective in GUI
        update_perspective()

        # Exit condition
        if keyboard.is_pressed("esc"):
            print("\nExiting simulation...")
            break

        time.sleep(0.01)  # Small delay to prevent excessive updates

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

# Run the keyboard input on a separate thread
threading.Thread(target=manual_control, daemon=True).start()

# Start the GUI loop
root.mainloop()
