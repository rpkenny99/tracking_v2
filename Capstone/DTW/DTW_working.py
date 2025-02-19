import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from concurrent.futures import ThreadPoolExecutor
import os
import glob

def load_multidimensional_data_as_dataframe(file_path):
    """
    Load a dataset from a specified text file path as a DataFrame.
    """ 
    try:
        # Update the line where you load the data
        data = pd.read_csv(file_path, sep='\s+', header=None)
        print(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def compute_dtw_dimension_fast(series1, series2):
    filtered_series1 = series1[~np.isnan(series1)]
    filtered_series2 = series2[~np.isnan(series2)]
    if len(filtered_series1) == 0 or len(filtered_series2) == 0:
        return np.nan
    return dtw.distance_fast(filtered_series1, filtered_series2)

def compute_dtw_per_dimension_parallel(array1, array2):
    num_dimensions = array1.shape[1]
    similarity_scores = [None] * num_dimensions

    def compute_dtw_for_dim(dim):
        series1 = array1[:, dim]
        series2 = array2[:, dim]
        return dim, compute_dtw_dimension_fast(series1, series2)

    with ThreadPoolExecutor() as executor:
        results = executor.map(compute_dtw_for_dim, range(num_dimensions))

    for dim, score in results:
        similarity_scores[dim] = score

    return similarity_scores

# def plot_dtw_alignment(series1, series2, reference_name, dimension):
#     """
#     Plot the DTW alignment path and distance/time matrix for two sequences.
#     """
#     # Compute DTW and alignment path
#     distance_matrix = dtw.distance_matrix([series1, series2], compact=False)
#     alignment_path = dtw.warping_path(series1, series2)
#
#     # Plot distance matrix
#     plt.figure(figsize=(8, 6))
#     plt.imshow(distance_matrix, origin='lower', cmap='viridis', interpolation='nearest')
#     plt.colorbar(label='Distance')
#     plt.plot(*zip(*alignment_path), color='red')  # Add alignment path
#     plt.title(f'Distance Matrix with Alignment Path (Dim {dimension}, {reference_name})')
#     plt.xlabel('Index in Sequence 1')
#     plt.ylabel('Index in Sequence 2')
#     plt.show()
#
#     # Plot aligned sequences
#     plt.figure(figsize=(10, 5))
#     plt.plot(series1, label='Input Sequence', alpha=0.7)
#     plt.plot(series2, label='Reference Sequence', alpha=0.7)
#     plt.title(f'Aligned Sequences (Dim {dimension}, {reference_name})')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.show()
def plot_dtw_alignment(series1, series2, reference_name, dimension):
    """
    Plot the DTW alignment path and distance/time matrix for two sequences.
    """
    # Compute DTW and alignment path
    distance = dtw.distance(series1, series2)  # Updated from distance_matrix
    alignment_path = dtw.warping_path(series1, series2)

    # Plot distance (using the direct DTW distance, not a matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(np.array([[distance]]), origin='lower', cmap='viridis', interpolation='nearest')  # Changed to handle 1x1 array
    plt.colorbar(label='Distance')
    plt.plot(*zip(*alignment_path), color='red')  # Add alignment path
    plt.title(f'Distance Matrix with Alignment Path (Dim {dimension}, {reference_name})')
    plt.xlabel('Index in Sequence 1')
    plt.ylabel('Index in Sequence 2')
    plt.show()

    # Plot aligned sequences
    plt.figure(figsize=(10, 5))
    plt.plot(series1, label='Input Sequence', alpha=0.7)
    plt.plot(series2, label='Reference Sequence', alpha=0.7)
    plt.title(f'Aligned Sequences (Dim {dimension}, {reference_name})')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # NEW: Plot warping distances over time
    warping_distances = [abs(series1[i] - series2[j]) for i, j in alignment_path]

    plt.figure(figsize=(8, 5))
    plt.plot(warping_distances, label='Warping Distance', color='purple')
    plt.title(f'Warping Distances Over Time (Dim {dimension}, {reference_name})')
    plt.xlabel('Step Along the Warping Path')
    plt.ylabel('Warping Distance')
    plt.legend()
    plt.grid()
    plt.show()

# def plot_combined_similarity_results(data_df1, data_df2, similarity_scores, reference_name):
#     num_dimensions = data_df1.shape[1]
#
#     # Heatmap of DTW Similarity Scores
#     plt.figure(figsize=(10, 6))
#     plt.imshow([similarity_scores], cmap="viridis", aspect="auto")
#     plt.colorbar(label="DTW Similarity Score")
#     plt.xlabel("Dimension")
#     plt.title(f"Heatmap of DTW Similarity Scores Across Dimensions ({reference_name})")
#     plt.show()
#
#     # Line Plot of Average DTW Similarity Over Time
#     avg_similarity_over_time = np.nanmean(
#         [data_df1.iloc[:, dim] - data_df2.iloc[:, dim] for dim in range(num_dimensions)],
#         axis=0
#     )
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(avg_similarity_over_time, color="blue", label="Average Difference")
#     plt.xlabel("Data Point")
#     plt.ylabel("Average DTW Difference")
#     plt.title(f"Average DTW Similarity Over Time ({reference_name})")
#     plt.legend()
#     plt.show()
#
#     # Parallel Coordinate Plot for both datasets
#     plt.figure(figsize=(12, 6))
#     for dim in range(num_dimensions):
#         plt.plot(data_df1[dim], label=f"Dimension {dim} - Dataset 1", alpha=0.7)
#         plt.plot(data_df2[dim], label=f"Dimension {dim} - {reference_name}", linestyle="--", alpha=0.7)
#     plt.xlabel("Data Points")
#     plt.ylabel("Value")
#     plt.title(f"Parallel Coordinate Plot of All Dimensions ({reference_name})")
#     plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), ncol=2, fontsize="small")
#     plt.show()
#
#     # Dimensional Difference Plot for each dimension
#     plt.figure(figsize=(12, 6))
#     for dim in range(num_dimensions):
#         difference = data_df1[dim] - data_df2[dim]
#         plt.plot(difference, label=f"Dimension {dim}", alpha=0.7)
#     plt.xlabel("Data Points")
#     plt.ylabel("Difference")
#     plt.title(f"Dimensional Difference Plot (Dataset 1 - {reference_name})")
#     plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), ncol=2, fontsize="small")
#     plt.show()
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_trajectory(data1, data2, reference_name):
    """
    Plot 3D trajectories of the input and reference data using position columns (X, Y, Z).
    """
    # Extract X, Y, Z columns for both datasets (assume first 3 columns are positions)
    x1, y1, z1 = data1[:, 0], data1[:, 1], data1[:, 2]
    x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory of the input (trainee)
    ax.plot(x1, y1, z1, label='Trainee Path', color='blue', alpha=0.7)

    # Plot the trajectory of the reference (expert)
    ax.plot(x2, y2, z2, label='Expert Path', color='green', linestyle='--', alpha=0.7)

    # Highlight start and end points
    ax.scatter(x1[0], y1[0], z1[0], color='blue', marker='o', s=100, label='Trainee Start')
    ax.scatter(x2[0], y2[0], z2[0], color='green', marker='o', s=100, label='Expert Start')
    ax.scatter(x1[-1], y1[-1], z1[-1], color='blue', marker='x', s=100, label='Trainee End')
    ax.scatter(x2[-1], y2[-1], z2[-1], color='green', marker='x', s=100, label='Expert End')

    # Set axis labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # Title and legend
    ax.set_title(f"3D Trajectory Comparison ({reference_name})")
    ax.legend()

    plt.show()

def plot_combined_similarity_results(data_df1, data_df2, similarity_scores, reference_name):
    num_dimensions = data_df1.shape[1]

    # Validate if both datasets have the same number of dimensions (columns)
    if data_df1.shape[1] != data_df2.shape[1]:
        raise ValueError("Input and reference datasets must have the same number of dimensions.")

    # Heatmap of DTW Similarity Scores
    plt.figure(figsize=(10, 6))
    plt.imshow([similarity_scores], cmap="viridis", aspect="auto")
    plt.colorbar(label="DTW Similarity Score")
    plt.xlabel("Dimension")
    plt.title(f"Heatmap of DTW Similarity Scores Across Dimensions ({reference_name})")
    plt.show()

    # Line Plot of Average DTW Similarity Over Time
    avg_similarity_over_time = np.nanmean(
        [data_df1.iloc[:, dim] - data_df2.iloc[:, dim] for dim in range(num_dimensions)],
        axis=0
    )

    plt.figure(figsize=(10, 5))
    plt.plot(avg_similarity_over_time, color="blue", label="Average Difference")
    plt.xlabel("Data Point")
    plt.ylabel("Average DTW Difference")
    plt.title(f"Average DTW Similarity Over Time ({reference_name})")
    plt.legend()
    plt.show()

    # Parallel Coordinate Plot for both datasets
    plt.figure(figsize=(12, 6))
    for dim in range(num_dimensions):
        plt.plot(data_df1[dim], label=f"Dimension {dim} - Dataset 1", alpha=0.7)
        plt.plot(data_df2[dim], label=f"Dimension {dim} - {reference_name}", linestyle="--", alpha=0.7)
    plt.xlabel("Data Points")
    plt.ylabel("Value")
    plt.title(f"Parallel Coordinate Plot of All Dimensions ({reference_name})")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), ncol=2, fontsize="small")
    plt.show()

    # Dimensional Difference Plot for each dimension
    plt.figure(figsize=(12, 6))
    for dim in range(num_dimensions):
        difference = data_df1[dim] - data_df2[dim]
        plt.plot(difference, label=f"Dimension {dim}", alpha=0.7)
    plt.xlabel("Data Points")
    plt.ylabel("Difference")
    plt.title(f"Dimensional Difference Plot (Dataset 1 - {reference_name})")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1), ncol=2, fontsize="small")
    plt.show()

# Main program
base_directory = input(
    "Please enter the path to the main directory containing 'input' and 'reference' subdirectories: ")
input_subdirectory = "input"
reference_subdirectory = "reference"

# Load the single input dataset
input_file_list = glob.glob(os.path.join(base_directory, input_subdirectory, "*.txt"))
if not input_file_list:
    print(f"No text files found in directory: {os.path.join(base_directory, input_subdirectory)}")
    exit()
else:
    # input_data_df = load_multidimensional_data_as_dataframe(input_file_list)
    input_data_df = load_multidimensional_data_as_dataframe(input_file_list[0])

    # Loop through each reference dataset in the "reference" subdirectory
    reference_file_list = glob.glob(os.path.join(base_directory, reference_subdirectory, "*.txt"))
    if not reference_file_list:
        print(f"No text files found in directory: {os.path.join(base_directory, reference_subdirectory)}")
        exit()
    else:
        for reference_file in reference_file_list:
            reference_data_df = load_multidimensional_data_as_dataframe(reference_file)
            reference_name = os.path.basename(reference_file)

            if input_data_df is not None and reference_data_df is not None:
                # Convert to numpy arrays for DTW computation
                array1 = input_data_df.to_numpy()
                array2 = reference_data_df.to_numpy()

                # Compute DTW similarity scores per dimension
                similarity_scores = compute_dtw_per_dimension_parallel(array1, array2)

                # Check if all similarity scores are NaN
                if np.all(np.isnan(similarity_scores)):
                    print(
                        f"\nThe similarity scores for {os.path.basename(input_file_list[0])} and {reference_name} are all NaN. Skipping analysis.")
                    continue

                # Calculate the average similarity score
                average_similarity = np.nanmean(similarity_scores)

                # Generate 3D trajectory plot
                print("\nGenerating 3D Trajectory Plot...")
                plot_3d_trajectory(array1, array2, reference_name)

                print(f"\nComparison with {reference_name}:")
                print("Similarity scores for each dimension:", similarity_scores)
                print("Average similarity score across all dimensions:", average_similarity)

                # Check if the average similarity score is zero (identical datasets)
                if np.isclose(average_similarity, 0):
                    print(f"\nThe datasets {os.path.basename(input_file_list[0])} and {reference_name} are identical. No difference to analyze.")
                    continue  # Skip generating graphs for identical datasets

                # Identify the three dimensions with the most difference (highest similarity scores)
                most_different_indices = np.argsort(similarity_scores)[-3:][::-1]  # Top 3 highest scores
                print(f"The three dimensions with the most difference (highest similarity scores) are: {most_different_indices}")

                # Generate plots for the three dimensions with the most difference
                for dimension in most_different_indices:
                    print(f"\nGenerating plots for Dimension {dimension}...")
                    plot_dtw_alignment(
                        array1[:, dimension],
                        array2[:, dimension],
                        reference_name,
                        dimension
                    )

                # Generate the combined similarity results graph
                plot_combined_similarity_results(input_data_df, reference_data_df, similarity_scores, reference_name)
            else:
                print(f"Failed to load dataset: {reference_name}")
