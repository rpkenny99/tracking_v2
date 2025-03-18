import numpy as np
from scipy.spatial.transform import Rotation as R
import feedback3


# alternate approach

def get_transformation_matrix(position, orientation):
    """Compute the transformation matrix from user's perspective."""
    rotation_matrix = R.from_euler('xyz', orientation, degrees=True).as_matrix()
    translation_vector = np.array(position).reshape(3, 1)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector.ravel()

    return transformation_matrix

def update_projection(ui_instance, position_source):
    """Updates the projection based on the user's position and orientation."""
    position, orientation = position_source()
    transformation_matrix = get_transformation_matrix(position, orientation)
    ui_instance.apply_transformation(transformation_matrix)

if __name__ == "__main__":
    queue = feedback3.Queue()
    ui_instance = feedback3.FeedbackUI("Left Vein", "Point A", work_queue=queue)

    while True:
        update_projection(ui_instance, mock_position_source)
