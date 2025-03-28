import pandas as pd
import numpy as np
from dtaidistance import dtw
from concurrent.futures import ThreadPoolExecutor
import os
import glob

def load_multidimensional_data_as_dataframe(file_path):
    """
    Load a dataset from a specified text file path as a DataFrame.
    """ 
    try:
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

def compute_dtw(expert_data="Capstone/DTW/mean_dtw.txt", 
                file_lock=None, 
                trainee_data="Capstone/Filter/filtered_data.txt"):
    # Load the trainee data
    with file_lock:
        input_data_df = load_multidimensional_data_as_dataframe(trainee_data)

    # Load the single reference expert data
    reference_data_df = load_multidimensional_data_as_dataframe(expert_data)
    reference_name = os.path.basename(expert_data)

    average_similarity = []

    if input_data_df is not None and reference_data_df is not None:
        array1 = input_data_df.to_numpy()
        array2 = reference_data_df.to_numpy()

        similarity_scores = compute_dtw_per_dimension_parallel(array1, array2)

        if np.all(np.isnan(similarity_scores)):
            print(
                f"\nThe similarity scores for {trainee_data} and {reference_name} are all NaN. Skipping analysis.")
        else:
            average_similarity.append(np.nanmean(similarity_scores))
            

            print(f"\nComparison with {reference_name}:")
            print("Similarity scores for each dimension:", similarity_scores)
            print("Average similarity score across all dimensions:", average_similarity)

            most_different_indices = np.argsort(similarity_scores)[-3:][::-1]
            print(f"The three dimensions with the most difference (highest similarity scores) are: {most_different_indices}")
    else:
        print(f"Failed to load either trainee or expert dataset.")

    return average_similarity
