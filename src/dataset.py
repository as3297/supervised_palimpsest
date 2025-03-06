import numpy as np
from read_data import read_subset_features
import tensorflow as tf
import os



def read_features(main_dir,folio_names,classes_dict,modalities):
    """
    Reads and aggregates feature data for multiple classes and folio names.

    Arguments:
    main_dir: The main directory containing the feature data.
    folio_names: A list of folder or file names to read features from.
    classes_dict: A dictionary mapping class names to corresponding class indices.
    modalities: Modalities to be included in the feature reading process.

    Returns:
    A dictionary where keys are class indices (from classes_dict) and values are concatenated feature arrays corresponding to those class indices.
    """
    features_dict = {}
    for class_name,class_idx in classes_dict.items():
        features_dict[class_idx] = []
        for folio_name in folio_names:
            features,xs,ys = read_subset_features(main_dir,folio_name,class_name,modalities,None)
            features_dict[class_idx].append(features)
        features_dict[class_idx] = np.concatenate(features_dict[class_idx],0)
    return features_dict

def stack_features_labels(features_dict):
    """
    Combines features and corresponding labels from a dictionary into single arrays.

    Parameters:
    features_dict: dict
        A dictionary where keys are class indices (or labels) and values are numpy arrays of features corresponding to each class.

    Returns:
    tuple
        A tuple containing two numpy arrays:
        - dataset: A 2D array where rows are concatenated feature vectors from all classes.
        - labels: A 1D array where each element is the class label corresponding to each feature vector in the dataset.
    """
    dataset = []
    labels = []
    for class_idx,features in features_dict.items():
            dataset.append(features)
            labels.append([class_idx]*features.shape[0])
    dataset = np.concatenate(dataset,0)
    labels = np.concatenate(labels,0)[:,np.newaxis]
    return dataset,labels

def create_tf_dataset(features_train,labels_train,features_val,labels_val):
    """

    """
    ds_train = tf.data.Dataset.from_tensor_slices({"features": features_train, "label": labels_train})
    ds_train = ds_train.shuffle(BUFFER_SIZE).repeat()

    ds_val = tf.data.Dataset.from_tensor_slices({"features": features_val, "label": labels_val})

    return ds_train, ds_val

def dataset(main_dir,folio_names_train,folio_names_val,class_names,modality):
    """
    Creates training and validation datasets by reading and processing feature data from the specified directories.

    Args:
        main_dir (str): Path to the main directory containing the dataset.
        folio_names_train (list): List of folder or file names for training data.
        folio_names_val (list): List of folder or file names for validation data.
        class_names (list): List of possible class names for classification tasks.
        modality (str): Specific data modality to be used (e.g., type of feature or data format).

    Returns:
        tuple: A tuple containing the processed training dataset and validation dataset.
    """
    features_dict_train = read_features(main_dir,folio_names_train,class_names,modality)
    features_dict_val = read_features(main_dir, folio_names_val, class_names, modality)

    dataset_val = stack_features_labels(features_dict_val)
    dataset_train = stack_features_labels(features_dict_train)
    return dataset_train,dataset_val


if __name__ == "__main__":
    main_data_dir = r"D:"
    palimpsest_name = r"Verona_msXL"
    base_data_dir = os.path.join(main_data_dir, palimpsest_name)
    folios_train = ["msXL_335v_b", ]
    classes_dict = {"undertext": 1}
    modalities = ["M"]
    features_dict_train = read_features(base_data_dir, folios_train, classes_dict, modalities)
    print("Features shape", features_dict_train[1].shape)
    print("Labels shape", features_dict_train[1].shape)