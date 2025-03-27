import numpy as np
from src.read_data import read_subset_features,read_subset_features_patches
import os



def read_features(main_dir,folio_names,classes_dict,modalities, box,win):
    """
    Reads and aggregates feature data for multiple classes and folio names.

    Arguments:
    main_dir: The main directory containing the feature data.
    folio_names: A list of folder or file names to read features from.
    classes_dict: A dictionary mapping class names to corresponding class indices.
    modalities: Modalities to be included in the feature reading process.
    win int
        if higher then zero then extract patches around the points with size (win+1,win+1)

    Returns:
    A dictionary where keys are class indices (from classes_dict) and values are concatenated feature arrays corresponding to those class indices.
    """
    features_dict = {}
    for class_name,class_idx in classes_dict.items():
        features_dict[class_idx] = []
        for folio_name in folio_names:
            if win>0:
                features,xs,ys = read_subset_features_patches(main_dir,folio_name,class_name,modalities,win,box)
            else:
                features,xs,ys = read_subset_features(main_dir,folio_name,class_name,modalities,box)
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

def dataset(main_dir,folio_names_train,folio_names_val,class_names,modality,win, debugging=False):
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
    if debugging:
        box = "val_bbox"
    else:
        box = None
    features_dict_train = read_features(main_dir,folio_names_train,class_names,modality,box,win)
    dataset_train = stack_features_labels(features_dict_train)
    if len(folio_names_val)>0:
        features_dict_val = read_features(main_dir, folio_names_val, class_names, modality,box,win)
        dataset_val = stack_features_labels(features_dict_val)
    else:
        dataset_val = None

    return dataset_train,dataset_val


if __name__ == "__main__":
    main_data_dir = r"D:"
    palimpsest_name = r"Verona_msXL"
    base_data_dir = os.path.join(main_data_dir, palimpsest_name)
    classes_dict = {"undertext": 1, "not_undertext": 0}
    modalities = ["M"]
    folios_train = ["msXL_335v_b"]
    win = 0
    box = None
    features_dict_train = read_features(base_data_dir, folios_train, classes_dict, modalities,box,win)
    print("Features shape", features_dict_train[1].shape)
    print("Labels shape", features_dict_train[1].shape)
    folios_train = ["msXL_335v_b", r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
                   "msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b",
                   "msXL_334v_b", "msXL_344r_b", "msXL_344v_b", r"msXL_315r_b"]

    features_dict_train = read_features(base_data_dir, folios_train, classes_dict, modalities,box,win)
    print("Features shape", features_dict_train[1].shape)
    print("Labels shape", features_dict_train[1].shape)