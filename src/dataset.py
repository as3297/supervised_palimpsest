import numpy as np
from src.read_data import read_subset_features
import os



def read_features(main_dir,folio_names,classes_dict,modalities, box):
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

def dataset(main_dir,folio_names_train,folio_names_val,class_names,modality, debugging=False):
    """
    Creates training and validation datasets by reading and processing feature data from the specified directories.

    Args:
        main_dir (str): Path to the main directory containing the dataset.
        folio_names_train (list): List of folder or file names for training data.
        folio_names_val (list): List of folder or file names for validation data.
        class_names (dict): Dict of possible class names for classification tasks.
        modality (str): Specific data modality to be used (e.g., type of feature or data format).

    Returns:
        tuple: A tuple containing the processed training dataset and validation dataset.
    """
    if debugging:
        box = "val"
    else:
        box = None
    features_dict_train = read_features(main_dir,folio_names_train,class_names,modality,box)
    dataset_train = stack_features_labels(features_dict_train)
    if len(folio_names_val)>0:
        features_dict_val = read_features(main_dir, folio_names_val, class_names, modality,box)
        dataset_val = stack_features_labels(features_dict_val)
    else:
        dataset_val = None

    return dataset_train,dataset_val


if __name__ == "__main__":
    main_data_dir = r"D:"
    palimpsest_name = r"Verona_msXL"
    main_dir = os.path.join(main_data_dir, palimpsest_name)
    classes_dict = {"undertext": 1, "not_undertext": 0}
    modalities = ["M"]
    folios_train = ["msXL_344v_b"]#["msXL_335v_b"]
    folios_val = ""#["msXL_344v_b"]
    win = 10
    box = "val"
    features_train,_ =  dataset(main_dir,folios_train,folios_val,classes_dict,modalities,win)
    print("Features train shape", features_train[0].shape)
    #print("Features val shape", features_val[0].shape)
    print("Labels shape", features_train[1].shape)
    unique_values, counts = np.unique(features_train[1], return_counts=True)
    # Print counts
    for value, count in zip(unique_values, counts):
        print(f"Value {value} appears {count} times")
