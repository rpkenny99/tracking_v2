import numpy as np
import cv2 as cv2

geometric_rotations = {
    "0": [180-116.565,      0,                      -112],
    "1": [180-116.565,      0,                      0],
    "2": [-19,              2*(180-116.565) - 15,   -58],
    "3": [-2*(180-116.565), 0,                      -102],
    "4": [19,               19,                     -(180-116.565)],
    "5": [-20,              (180-116.565),          120],
    "6": [28,                (180-116.565),          45],
    "8": [39,               2*(180-116.565),        -98],
    "9": [42,               2*(180-116.565),        115],
    "11": [-57,             -38,                    12]
}

def rotation_around_y(pose, angle):
    angle = np.deg2rad(angle)
    # Define the axis for rotation (for instance, rotate around the y-axis)
    rotation_axis = np.array([0, 1, 0])  # Rotate around the y-axis

    # Create the additional rotation vector for 64 degrees around the specified axis
    additional_rotation_vector = rotation_axis * angle

    # Convert additional rotation vector to rotation matrix
    additional_rotation_matrix, _ = cv2.Rodrigues(additional_rotation_vector)

    # Combine the rotation matrices by matrix multiplication
    combined_rotation_matrix = pose @ additional_rotation_matrix

    # Use the combined rotation matrix
    return combined_rotation_matrix

def rotation_around_x(pose, angle):
    angle = np.deg2rad(angle)
    # Define the axis for rotation (for instance, rotate around the y-axis)
    rotation_axis = np.array([1, 0, 0])  # Rotate around the y-axis

    # Create the additional rotation vector for 64 degrees around the specified axis
    additional_rotation_vector = rotation_axis * angle

    # Convert additional rotation vector to rotation matrix
    additional_rotation_matrix, _ = cv2.Rodrigues(additional_rotation_vector)

    # Combine the rotation matrices by matrix multiplication
    combined_rotation_matrix = pose @ additional_rotation_matrix

    # Use the combined rotation matrix
    return combined_rotation_matrix


def rotation_around_z(pose, angle):
    angle = np.deg2rad(angle)
    # Define the axis for rotation (for instance, rotate around the y-axis)
    rotation_axis = np.array([0, 0, 1])  # Rotate around the y-axis

    # Create the additional rotation vector for 64 degrees around the specified axis
    additional_rotation_vector = rotation_axis * angle

    # Convert additional rotation vector to rotation matrix
    additional_rotation_matrix, _ = cv2.Rodrigues(additional_rotation_vector)

    # Combine the rotation matrices by matrix multiplication
    combined_rotation_matrix = pose @ additional_rotation_matrix

    # Use the combined rotation matrix
    return combined_rotation_matrix

def rotate_marker_1(pose):
    pose = rotation_around_x(pose,  -(180-116.565))
    pose = rotation_around_z(pose,  198)
    return pose

def rotate_marker_2(pose):
    pose = rotation_around_y(pose, -(180-116.565))
    pose = rotation_around_z(pose, 216)
    return pose

def rotate_marker_3(pose):
    pose = rotation_around_y(pose, -(180-116.565))
    pose = rotation_around_z(pose,  -216)
    return pose

def rotate_marker_4(pose):
    pose = rotation_around_y(pose, (180-116.565))
    pose = rotation_around_z(pose,  -108)
    return pose

def rotate_marker_5(pose):
    pose = rotation_around_x(pose, (180-116.565))
    pose = rotation_around_z(pose, 90)
    return pose

def rotate_marker_6(pose):
    pose = rotation_around_x(pose, (180-116.565 + 54))
    pose = rotation_around_z(pose, -90)
    return pose

def rotate_marker_7(pose):
    pose = rotation_around_y(pose, -(180-116.565 + 54))
    pose = rotation_around_z(pose, 108)
    return pose

def rotate_marker_8(pose):
    pose = rotation_around_x(pose, (180-116.565 + 54))
    pose = rotation_around_z(pose,  -234)
    return pose

def rotate_marker_9(pose):
    pose = rotation_around_x(pose, -(180-116.565 + 54))
    pose = rotation_around_z(pose,  -126)
    return pose

def rotate_marker_14(pose):
    pose = rotation_around_x(pose, -(180-116.565 + 54))
    pose = rotation_around_z(pose, -198)
    return pose

def rotate_marker_19(pose):
    pose = rotation_around_x(pose, 180)
    pose = rotation_around_z(pose, 198)
    return pose