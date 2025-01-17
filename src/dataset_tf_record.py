from random import shuffle

import numpy as np
import tensorflow as tf
from read_data import read_msi_image_object, read_x_y_coords
from msi_data_as_array import PointfromMSI_PIL,PointsfromMSI_PIL
from label_noise_clean import knn_clean_ut


def create_dataset_coords(coords, labels,main_paths, palimpsest_names,shuffle=True):
    """
    Creates a TensorFlow dataset from the provided input data.

    Args:
        coords: List or array-like structure containing coordinates data.
        labels: List or array-like structure containing labels corresponding to the data.
        main_paths: List or array-like structure containing file paths or main path information.
        palimpsest_names: List or array-like structure containing names or identifiers related to palimpsests.

    Returns:
        A TensorFlow Dataset object containing the provided data as dictionary elements. Each element includes:
            - "coords": The provided coordinates.
            - "labels": The provided labels.
            - "main_paths": The provided main paths.
            - "palimpsest_names": The provided palimpsest names.
            - "pixel_values": A placeholder list of None values with a length equal to the number of labels.
    """
    #tf_labels = tf.convert_to_tensor(labels,name="labels")
    #tf_coords = tf.convert_to_tensor(coords,name="coords")
    #tf_main_paths = tf.convert_to_tensor(main_paths,name="main_paths")
    #tf_palimpsest_names = tf.convert_to_tensor(palimpsest_names,name="palimpsest_names")
    #tf_pixel_values = tf.zeros((len(labels),),name="pixel_values")

    #dataset = tf.data.Dataset.from_tensor_slices({"coords": tf_coords, "labels": tf_labels, "main_paths": tf_main_paths, "palimpsest_names": tf_palimpsest_names,
    # "pixel_values": tf_pixel_values})

    dataset = tf.data.Dataset.from_tensor_slices({"coords": coords, "labels": labels, "main_paths": main_paths,
                                                  "palimpsest_names": palimpsest_names,"pixel_values": [None]*len(labels)})
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))
    return dataset


def get_pixelvalues_py(coords, main_paths, palimpsest_names):
    """
    Extract pixel values from multispectral images (MSI) based on given coordinates and file names.

    Parameters:
    coords: Tensor representing the coordinates for which pixel values are to be extracted. Converted to NumPy array for processing.
    main_paths: Tensor containing the path to the main directory. The first path in the tensor is decoded to a UTF-8 string.
    palimpsest_names: Tensor containing the filenames of the MSI images. Converted and decoded to a list of UTF-8 strings.

    Returns:
    A Tensor containing the pixel values corresponding to the given coordinates and filenames.

    Description:
    This function reads multispectral image (MSI) objects specified by their filenames and a main directory path. Given a set of coordinates, it extracts the pixel values from the corresponding locations in the images. The function first loads the MSI images using a helper function, storing them in a dictionary keyed by their filenames to avoid redundant file reads. For each filename and coordinate pair, it retrieves the corresponding pixel value and stores it in a list, which is finally converted into a Tensor and returned.
    """
    coords = coords.numpy()
    main_path = main_paths.numpy()[0].decode('utf-8')
    palimpsest_names = [palimpsest_names.decode('utf-8') for palimpsest_names in palimpsest_names.numpy()]
    pixel_values = []
    modality = "M"
    pal_obj_dict = {palimpsest_name:read_msi_image_object(main_path, palimpsest_name, modality) for palimpsest_name in set(palimpsest_names)}
    for coord, palimpsest_name in zip(coords,palimpsest_names):
        palimpsest_obj = pal_obj_dict[palimpsest_name]
        pixel_value = PointfromMSI_PIL(palimpsest_obj,coord).point
        pixel_values.append(pixel_value)
    pixel_values = tf.convert_to_tensor(pixel_values)
    return pixel_values


def get_pixelvalues_tf(coords, main_paths, palimpsest_paths):
    """
    Fetches pixel values from images using TensorFlow.

    This function wraps around a Python function `get_pixelvalues_py` and enables its use
    in a TensorFlow pipeline. It extracts pixel values from specified image file paths
    based on the provided coordinates. The function returns the computed values in the
    TensorFlow-compatible tf.float64 format.

    Parameters:
    coords: Tensor or array-like
        The coordinates used to fetch specific pixel values from the images.
    main_paths: Tensor or array-like
        The paths to the main set of images from which pixel values will be extracted.
    palimpsest_paths: Tensor or array-like
        The paths to the palimpsest set of images used in coordination with the main_paths for processing.

    Returns:
    Tensor
        A TensorFlow tensor of type tf.float64 containing the extracted pixel values.
    """
    return tf.py_function(func=get_pixelvalues_py, inp=[coords, main_paths, palimpsest_paths], Tout=tf.float64)

def processing(element):
    """
    Processes an input dictionary by extracting coordinates, main paths, and palimpsest names, computes pixel values, and updates the dictionary.

    Parameters:
    element (dict): A dictionary containing:
        - "coords" (list/array): Coordinates for pixel value computation.
        - "main_paths" (list): Main paths associated with the input.
        - "palimpsest_names" (list): Names or paths used as palimpsests.

    Returns:
    tuple: A tuple containing:
        - pixel_values: The computed pixel values based on input data.
        - element["labels"]: Labels associated with the processed element.
    """
    coords = element["coords"]
    main_paths = element["main_paths"]
    palimpsest_paths = element["palimpsest_names"]
    pixel_values = get_pixelvalues_tf(coords, main_paths, palimpsest_paths)
    element["pixel_values"] = pixel_values
    return pixel_values, element["labels"]

def processing_test(element):
    """
    Processes an input dictionary by extracting coordinates, main paths, and palimpsest names, computes pixel values, and updates the dictionary.

    Parameters:
    element (dict): A dictionary containing:
        - "coords" (list/array): Coordinates for pixel value computation.
        - "main_paths" (list): Main paths associated with the input.
        - "palimpsest_names" (list): Names or paths used as palimpsests.

    Returns:
    tuple: A tuple containing:
        - pixel_values: The computed pixel values based on input data.
        - element["labels"]: Labels associated with the processed element.
    """
    coords = element["coords"]
    main_paths = element["main_paths"]
    palimpsest_paths = element["palimpsest_names"]
    pixel_values = get_pixelvalues_tf(coords, main_paths, palimpsest_paths)
    element["pixel_values"] = pixel_values
    return pixel_values, element["labels"], element["coords"], element["main_paths"], element["palimpsest_names"]

def read_coords(main_dir, pal_name, modality):
    """
    Reads and processes coordinates for specified palimpsest.

    Parameters:
    main_dir: Path to the main directory containing the palimpsest folder.
    pal_name: Name of the palimpsest
    modality: Specific imaging modality to be used, "M"-reflectance, "W" - flourescence

    Returns:
    im_pil_ob: The MSI image object read using the specified parameters.
    coords_ut: Numpy array containing x and y coordinates for "undertext".
    coords_not_ut: Numpy array containing x and y coordinates for "not undertext".
    """
    im_pil_ob = read_msi_image_object(main_dir, pal_name, modality)
    xs_ut, ys_ut = read_x_y_coords(main_dir, pal_name, "undertext", im_pil_ob, None)
    xs_not_ut, ys_not_ut = read_x_y_coords(main_dir, pal_name, "not_undertext", im_pil_ob, None)
    coords_ut = np.column_stack((xs_ut, ys_ut))
    coords_not_ut = np.column_stack((xs_not_ut, ys_not_ut))
    if len(coords_ut) == 0 or len(coords_not_ut) == 0:
        raise ValueError(f"No valid coordinates found for palette {pal_name}.")
    return im_pil_ob, coords_ut, coords_not_ut

def filter_ut_coords_from_noise(im_pil_ob,coords_ut,coords_not_ut):
    """
    Filters coordinates from noisy points using K-Nearest Neighbors (KNN) cleaning process.

    Args:
        im_pil_ob: PIL image object representing the MSI image.
        coords_ut: List of undertext coordinates to be filtered.
        coords_not_ut: List of non-undertext coordinates used as a reference for filtering.

    Returns:
        A tuple containing:
            - Filtered undertext coordinates (cleaned `coords_ut`).
            - Original non-undertext coordinates (`coords_not_ut`).

    The method uses the PointsfromMSI_PIL class to retrieve features for both ut and non-ut coordinates,
    then applies the KNN algorithm to clean ut points based on their distance to non-ut points.
    """
    points_object = PointsfromMSI_PIL(pil_msi_obj=im_pil_ob, points_coord=coords_ut)
    features_ut = points_object.points
    points_object = PointsfromMSI_PIL(pil_msi_obj=im_pil_ob, points_coord=coords_not_ut)
    features_not_ut = points_object.points
    coords_ut = knn_clean_ut(features_ut, features_not_ut,K=4,coord_ut=coords_ut)
    return coords_ut, coords_not_ut

def create_oversampled_dataset(batch_size,pal_names,main_dir,modality,shuffle,filter_label_noise, test_on_subset):
    """
    Creates an oversampled TensorFlow dataset based on given input parameters, with options to filter label noise and shuffle data.

    Parameters:
    - batch_size (int): Size of the batches for the final dataset.
    - pal_names (list of str): List of palette names used to extract coordinates.
    - main_dir (str): Main directory path containing the data.
    - modality (str): Modality type (e.g., type of data format or category).
    - shuffle (bool): Flag indicating whether to shuffle the dataset or not.
    - filter_label_noise (bool): Flag to enable or disable filtering of noisy labeled data.

    Returns:
    - tensorflow.data.Dataset: An oversampled and optionally shuffled preprocessed dataset ready for training.

    Details:
    - Reads coordinates for `ut` (e.g., a specific label class) and `nonut` (other label class) from the main directory for given palette names.
    - Optionally filters out noisy labels for both `ut` and `nonut`.
    - Combines coordinates, labels, main directory paths, and palette names into individual datasets for `ut` and `nonut` classes.
    - Creates a resampled dataset using weights specified for the classes.
    - Shuffles the dataset if enabled, batches the data using the specified batch size, prefetches for optimized pipeline performance, and applies a processing map function in parallel.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(pal_names, list) or not all(isinstance(p, str) for p in pal_names):
        raise ValueError("pal_names must be a list of strings.")
    if not isinstance(main_dir, str):
        raise ValueError("main_dir must be a string.")
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle must be a boolean.")

    coords_ut = []
    coords_nonut = []
    labels_ut = []
    labels_nonut = []
    main_paths_ut = []
    main_paths_nonut = []
    pal_paths_ut = []
    pal_paths_nonut = []
    for pal_name in pal_names:
        #read coordinates and msi image object
        im_pil_ob, coord_ut, coord_nonut = read_coords(main_dir,pal_name,modality)
        print(f"Extracted coordinates from {pal_name}")
        #find number of undertext and not undertext samples
        nb_ut_samples = len(coord_ut)
        nb_nonut_samples = len(coord_nonut)
        if test_on_subset:
            coord_ut, coord_nonut = coord_ut[:min(batch_size,nb_ut_samples)], coord_nonut[:min(batch_size,nb_nonut_samples)]
            nb_ut_samples = len(coord_ut)
            nb_nonut_samples = len(coord_nonut)
        # filter ut_coord using Knn algorithm
        if filter_label_noise:
            coord_ut, coord_nonut = filter_ut_coords_from_noise(im_pil_ob, coord_ut, coord_nonut)
            print(f"Filtered coordinates from {pal_name}")
        #add coordinates from current palimpsest to the list of coordinates from all palimpsests
        print("Number of ut samples {}, number of nonut samples {}".format(nb_ut_samples,nb_nonut_samples))
        coords_ut.append(coord_ut)
        coords_nonut.append(coord_nonut)
        #add labels to undertext and non_undertext labels lists
        labels_ut.extend([1] * nb_ut_samples)
        labels_nonut.extend([0] * nb_nonut_samples)
        #add main_path and palimpsest name to undertext and non_undertext main_paths and pal_paths lists
        #this is done that every element of dataset had access to the full path of the palimpsest
        main_paths_ut.extend([main_dir] * nb_ut_samples)
        main_paths_nonut.extend([main_dir] * nb_nonut_samples)
        pal_paths_ut.extend([pal_name] * nb_ut_samples)
        pal_paths_nonut.extend([pal_name] * nb_nonut_samples)
    coords_ut = np.concatenate(coords_ut,axis=0)
    coords_nonut = np.concatenate(coords_nonut,axis=0)

    dataset_coords_ut = create_dataset_coords(coords_ut, labels_ut, main_paths_ut, pal_paths_ut,shuffle)
    dataset_coords_nonut = create_dataset_coords(coords_nonut, labels_nonut, main_paths_nonut, pal_paths_nonut,shuffle)
    resampled_ds = tf.data.Dataset.sample_from_datasets([dataset_coords_ut, dataset_coords_nonut], weights=[0.5, 0.5])
    resampled_ds = resampled_ds.batch(batch_size)
    if test_on_subset:
        resampled_ds = resampled_ds.map(processing_test,num_parallel_calls=4)
    else:
        resampled_ds = resampled_ds.map(processing, num_parallel_calls=4)
    resampled_ds = resampled_ds.prefetch(buffer_size=4)
    return resampled_ds

#TODO: test the actual values from dataset
def create_oversampled_dataset_test():
    batch_size = 256
    main_dir = r"D:\Verona_msXL"
    pal_names = ["msXL_315r_b","msXL_319r_b"]
    modality = "M"
    resample_ds = create_oversampled_dataset(batch_size,pal_names,main_dir,modality,False,False,True)
    print(
        f"The dataset contains {resample_ds.cardinality().numpy()} elements, with {resample_ds.element_spec} as element spec."
    )
    for element in resample_ds:
        print(element)
        pixel_values = element[0].numpy()
        coords = element[2].numpy()
        pal_names = element[4].numpy()
        for pixel_value, coord, pal_name in zip(pixel_values, coords, pal_names):
            im_pil_ob = read_msi_image_object(main_dir, pal_name.decode("utf-8"), modality)
            points_object = PointfromMSI_PIL(pil_msi_obj=im_pil_ob, point_coord= list(coord))
            features = points_object.point
            print(pixel_value)
            print(features)
            if np.any(pixel_value != features):
                AssertionError("Pixel values do not match")


if __name__ == "__main__":
    create_oversampled_dataset_test()