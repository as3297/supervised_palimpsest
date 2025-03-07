import json
import numpy as np
from decimal import Decimal
import os
import pickle

def save_pickle(fpath,data):
    # Save dictionary to a file
    with open(fpath, "wb") as file:
        pickle.dump(data, file)

def read_pickle(fpath):
    # Load dictionary from a file
    with open(fpath, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data

def save_json(fpath,d):
    """
    Save json file
    fpath - path to json file, should end with .json
    d - dictionary to save
    """
    # Write the dictionary to a JSON file
    with open(fpath, 'w') as json_file:
        json.dump(d, json_file, indent=4)

def extend_json(fpath,d):
    """
    Extend dictionary in json file
    fpath - path to json file, should end with .json
    d - dictionary to save
    """
    d_old = read_json(fpath)
    d.update(d_old)
    # Write the dictionary to a JSON file
    with open(fpath, 'w') as json_file:
        json.dump(d, json_file, indent=4)

def read_json(fpath):
    """
    Reads a JSON file from the given path.

    Parameters:
    fpath (str): Path to the JSON file.

    Returns:
    dict: The parsed dictionary if the file contains valid JSON.
          An empty dictionary if the file is empty or not valid JSON.
    """
    try:
        # Check if file is empty
        if os.stat(fpath).st_size == 0:
            print(f"Warning: The file '{fpath}' is empty.")
            return {}

        # Attempt to load JSON
        with open(fpath, 'r') as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        print(f"Error: The file '{fpath}' does not exist.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{fpath}'. Detail: {e}")
        return {}

def read_band_list(fpath,modalities):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    bands = []
    for line in lines:
        bands.append(line.strip("\n"))
    if not modalities is None:
        bands = sublist_of_bands(bands,modalities)
    return bands

def sublist_of_bands(bands,modalities=["M"]):
  bands_subset = []
  for band in bands:
      for modality in modalities:
        if modality in band:
          bands_subset.append(band)
  return bands_subset

def order_band_list(bands):
    ordered_bands = []
    list_of_band_idxs = []
    for band in bands:
        list_of_band_idxs.append(int(band[-4:-2]))

    for band_idx in sorted(list_of_band_idxs):
        for band_name in bands:
            print("Band name:{},idx:{}".format(band_name, band_name[-4:-2]))
            if int(band_name[-4:-2]) == band_idx:
                ordered_bands.append(band_name + "\n")
                break
    return ordered_bands

def generate_coord_inside_bbox(x1,y1,width,height):
    points_coords = []
    for j in range(height):
        for i in range(width):
            points_coords.append([i+x1,j+y1])
    return points_coords

debugging_dict = {
    "val_bbox": {
        "x1": 60,
        "y1": 1090,
        "x2": 1000,
        "y2": 1500
    },
    "train_bbox": {
        "x1": 60,
        "y1": 1090,
        "x2": 1000,
        "y2": 1500
    },
    "test_bbox": {
        "x1": 60,
        "y1": 1090,
        "x2": 1000,
        "y2": 1500
    }
}

def read_split_box_coord(split_name,d_bboxs):
    """
    Read bbox coords from json file according to dataset split name
    :param split_name: str,  "train", "test", "val"
    :param d_bboxs: dictionary for the class with data split boxes
    :return:
    list, [x1,y1,x2,y2]
    """
    bbox = []
    for bbox_name,d_bbox in d_bboxs.items():
        if split_name in bbox_name:
            bbox = [d_bbox["x1"],d_bbox["y1"],d_bbox["x2"],d_bbox["y2"]]
    if len(bbox)==0:
        raise IOError("Split name \"{}\" does not correspond any of the box names \"{}\"".format(split_name,d_bboxs.keys()))
    return bbox

def is_decimal_string(value):
    """
    Determines if a given value can be interpreted as a decimal number.

    This function attempts to convert the provided value into a decimal.Decimal object. If the conversion is successful, the function returns True, indicating that the input can be interpreted as a decimal number. If the conversion raises an exception, the function returns False.

    Parameters:
    value: The input value to check.

    Returns:
    bool: True if the value can be converted to a decimal.Decimal, otherwise False.
    """
    try:
        Decimal(value)  # Attempt to convert to Decimal
        return True
    except:
        return False

def convert_float_in_dict(dict):
    """
    Converts string representations of decimal numbers in a dictionary to float.

    This function iterates over all key-value pairs in the input dictionary.
    If a value is a string and can be interpreted as a decimal number, it will
    convert that string into a float and update the dictionary in-place.

    Parameters:
    dict (dict): A dictionary containing key-value pairs where some values may
                 be strings representing decimal numbers.

    Returns:
    dict: The modified dictionary with all decimal strings converted to floats.
    """
    for key, value in dict.items():
        if isinstance(value, (str, bool)) or value is None:
            continue
        else:
            dict[key] = float(value)
    return dict

def load_confusion_matrix(restore_path):
    """Load confusion matrix"""
    return np.load(os.path.join(restore_path, "baseline_confusion_matrix.npy"), allow_pickle=True)

def calculate_confusion_matrix(model,model_dir,X_train,Y_train,batch_size,nb_classes):
    """Calculate and save confusion_matrix"""
    ybaseline_predict = model.predict(X_train, batch_size=batch_size)
    ybaseline_predict = np.argmax(ybaseline_predict, axis=-1)
    baseline_confusion = np.zeros((nb_classes, nb_classes))
    for n, p in zip(Y_train, ybaseline_predict):
        baseline_confusion[p, n] += 1.
    np.save(os.path.join(model_dir, "baseline_confusion_matrix.npy"), baseline_confusion)
    return baseline_confusion