import numpy as np
from sklearn.neighbors import NearestNeighbors
from copy import deepcopy

def knn_clean_ut(features_ut, features_non_ut, K, coord_ut):
    """
    Filters and removes undertext coordinates that are closer to non-undertext points
    based on the k-nearest neighbors (KNN) algorithm.

    Parameters:
    ut_features: numpy.ndarray
        Features or data points associated with undertext data.
    non_ut_features: numpy.ndarray
        Features or data points associated with non-undertext data.
    K: int
        Number of nearest neighbors to consider.
    coord_ut: numpy.ndarray
        Coordinates of undertext points to be filtered.

    Returns:
    numpy.ndarray
        Filtered coordinates of undertext points that are farther away
        from non-undertext points based on the mean distance threshold.
    """
    nn_model = NearestNeighbors(n_neighbors=K, algorithm='kd_tree')
    nn_model.fit(features_non_ut)
    distances_non_undertext, _ = nn_model.kneighbors(features_non_ut)
    distances_non_undertext = distances_non_undertext[:, 1:]
    means_distances_non_undertext = np.mean(distances_non_undertext, axis=1)
    mean_distance_non_undertext = np.mean(means_distances_non_undertext, axis=None)
    distances_undertext, _ = nn_model.kneighbors(features_ut, n_neighbors=K - 1)
    means_of_k_neighbors_from_nonut_to_ut = np.mean(distances_undertext, axis=1)
    ut_points_cleaned_mask = means_of_k_neighbors_from_nonut_to_ut > mean_distance_non_undertext
    return coord_ut[ut_points_cleaned_mask,:]


def knn_clean_ut_test():
    coords_ut = np.array([[1,1],[1,2],[1,3]])
    features_ut = np.array([[0.1,0.2,0.3],[1,2,3],[1,3,4]])
    features_not_ut = deepcopy(features_ut[0])+np.random.normal(0,0.1, (100,3))
    result_coords_ut = knn_clean_ut(features_ut, features_not_ut, 4, coords_ut)
    # Raise a generic exception with a custom message
    if np.any(result_coords_ut != coords_ut[1:]):
        raise Exception("Something went wrong")


if __name__=="__main__":
    knn_clean_ut_test()

