import numpy as np
from pixel_coord import points_coord_in_bbox
from msi_data_as_array import PointsfromMSI_PIL,FullImageFromPILImageCube
from pil_image_cube import ImageCubePILobject
from util import read_band_list, read_json, extend_json, debugging_dict, read_split_box_coord, save_json
from copy import deepcopy
import os
from matplotlib import pyplot as plt
osp = os.path.join


def load_data_for_training(model_path: str, modalities: list, base_data_dir,folios,ut_mask_file,nonut_mask_file, debugging=False):
    """
    Load features and labels from multiple folios
    """

    band_list_path = osp(base_data_dir, "band_list.txt")
    bands = read_band_list(band_list_path,modalities)
    bbox_dicts = []
    dataset = {}
    for idx, folio_name in enumerate(folios):
        print("Folio name ",folio_name)
        bbox_fpath = osp(base_data_dir, folio_name, "dataset_split.json")
        bbox_dict = read_json(bbox_fpath)
        if debugging:
            bbox_dict = debugging_dict
        bbox_dicts.append(bbox_dict)

        folio_dataset = load_data_for_training_from_folio(base_data_dir, folio_name, bands, bbox_dict,ut_mask_file,nonut_mask_file)

        #test_load_data_for_training_from_folio(main_path, folio_name, bands, bbox_dict,folio_dataset)
        if idx == 0:
            for subset, subset_data in folio_dataset.items():
                dataset[subset] = [subset_data[0], subset_data[1]]
        else:
            for subset, subset_data in folio_dataset.items():
                dataset[subset][0] = np.concatenate([dataset[subset][0], subset_data[0]], axis=0)
                dataset[subset][1] = np.concatenate([dataset[subset][1], subset_data[1]], axis=0)
        if len(model_path)>0:
            save_data_parameters(model_path, modalities, bbox_dicts, folios,ut_mask_file,nonut_mask_file)
    return dataset["train_ut"],dataset["train_nonut"],dataset["val_ut"],dataset["val_nonut"]


def save_data_parameters(save_path, modalities: list, bbox_dicts: list, folios: list, ut_mask_file, nonut_mask_file):
    d = {}
    d["folios"] = folios
    d["modalities"] = modalities
    d["coord_boxs"] = bbox_dicts
    d["ut_mask_file"] = ut_mask_file
    d["nonut_mask_file"] = nonut_mask_file
    save_path = osp(save_path, "dataset_par.json")
    if not os.path.exists(save_path):
        save_json(save_path, d)
    else:
        extend_json(save_path, d)



def load_data_for_training_from_folio(main_path, folio_name, bands, bbox_dict,ut_mask_file,nonut_mask_file):
    """
    Load features and labels from one folio
    :return:
    [],[],[],[] - first list train data [features,labels] for undertext,
              second list train data [features,labels] for nonundertext,
              third list validation data [features,labels] for  undertext,
              forth list validation data [features,labels] for  nonundertext

    """

    image_dir_path = osp(main_path, folio_name)

    ut_mask_path = osp(image_dir_path, "mask", folio_name +"-" + ut_mask_file + r".png")
    nonut_mask_path = osp(image_dir_path, "mask", folio_name +"-" + nonut_mask_file + r".png")
    im_msi_pil_ob = ImageCubePILobject(main_path, folio_name, bands, 0)
    trainset_ut, trainset_nonut = read_point_class_ut_nonut_split(bbox_dict, im_msi_pil_ob,
                                                                  ut_mask_path, nonut_mask_path,  "train")
    valset_ut, valset_nonut = read_point_class_ut_nonut_split(bbox_dict, im_msi_pil_ob,
                                                              ut_mask_path, nonut_mask_path, "val")

    return {"train_ut": trainset_ut, "train_nonut": trainset_nonut, "val_ut": valset_ut, "val_nonut": valset_nonut}


def shuffle_between_epoch(set_ut, set_nonut):
    """
    Shuffles elements between two sets, set_ut and set_nonut, balancing the elements across epochs.

    Parameters:
    - set_ut: A set containing elements for class 'ut'.
    - set_nonut: A set containing elements for class 'nonut'.

    Returns:
    - A tuple containing two lists:
      1. A list with shuffled elements from set_ut, interleaved across epochs.
      2. A list with shuffled elements from set_nonut, interleaved across epochs.

    Raises:
    - ValueError: If either set_ut or set_nonut is empty.
    """
    set_ut = shuffle_dataset_split(set_ut)
    set_nonut = shuffle_dataset_split(set_nonut)
    dataset = equalize_nb_dataset_points(set_ut, set_nonut)
    dataset = shuffle_dataset_split(dataset)
    nb_samples = len(dataset[1])
    return dataset, nb_samples


def test_shuffle_between_epoch():
    # Create dummy datasets
    set_ut = [np.array([[0, 1], [1, 0], [2, 1]]), np.array([1, 1, 1])]
    set_nonut = [np.array([[1, 2], [2, 0], [1, 1], [3,3],[4,4]]), np.array([0, 0, 0,0,0])]

    # Call the function
    dataset, nb_samples = shuffle_between_epoch(set_ut, set_nonut)

    # Check shapes
    assert dataset[0].shape == (2*min(len(set_ut[0]), len(set_nonut[0])), set_ut[0].shape[1]), "Features shape mismatch"
    assert dataset[1].shape == (2*min(len(set_ut[1]), len(set_nonut[1])),), "Labels shape mismatch"
    assert nb_samples == len(dataset[1]), "Number of samples mismatch"



def read_points_class(fpath_image_mask,im_msi_pil_ob,bbox,label):
    """
    Reads the coordinates and associated labels of points within a bounding box from an image mask and a multispectral image.

    Parameters:
    fpath_image_mask : str
        The file path to the image mask.
    im_msi_pil_ob : PIL.Image.Image
        The PIL object representing the multispectral image.
    bbox : tuple
        The bounding box within which points are to be retrieved.
    label : int
        The label to assign to all retrieved points.

    Returns:
    tuple
        A tuple containing:
        - features (ndarray): The features of the points retrieved.
        - labels (ndarray): The labels for each point.
        - coords (list): The coordinates of the points retrieved.
    """
    xs, ys, nb_coords = points_coord_in_bbox(fpath_image_mask, bbox)
    labels = np.array([label]*nb_coords)
    coords = list(zip(xs,ys))
    points_object = PointsfromMSI_PIL(im_msi_pil_ob, coords)
    features = points_object.points
    return features,labels,coords


def read_point_class_ut_nonut_split(bbox_dict,im_msi_pil_ob,ut_mask_path,nonut_mask_path,split_name):
    """
    Read features and labels for two opposites classes of datasets
    :param dict_bbox_ut:
    :param dict_bbox_nonut:
    :param im_msi_pil_ob:
    :param ut_mask_path:
    :param nonut_mask_path:
    :param max_vals:
    :return:
    [],[] - fist list is a list of [features,labels] for undertext, second - of [features,labels] for nonundertext
    """
    bbox = read_split_box_coord(split_name, bbox_dict)
    feat_ut, labels_ut,coords_ut = read_points_class(ut_mask_path, im_msi_pil_ob, bbox, 1)
    feat_nonut, labels_nonut,coords_nonut = read_points_class(nonut_mask_path, im_msi_pil_ob, bbox, 0)
    return [feat_ut,labels_ut,coords_ut],[feat_nonut,labels_nonut,coords_nonut]

def shuffle_dataset_split(data_subset,show_split_idxs=False):
    """

    Shuffles the given dataset split and optionally returns the shuffled indices.

    Args:
        data_subset (tuple): A tuple containing two elements, where the first element is the dataset features and the second element is the dataset labels.
        show_split_idxs (bool): Flag to determine whether to return the shuffled indices along with the data. Defaults to False.

    Returns:
        list: A list containing two elements, the first is the shuffled dataset features and the second is the shuffled dataset labels.
        tuple: If `show_split_idxs` is True, returns a tuple where the first element is the shuffled data (a list) and the second element is an array of shuffled indices.
    """
    nb_points = len(data_subset[1])
    split_idxs = np.arange(nb_points)
    np.random.shuffle(split_idxs)
    if show_split_idxs:
        return [data_subset[0][split_idxs,:], data_subset[1][split_idxs]], split_idxs
    else:
        return [data_subset[0][split_idxs, :], data_subset[1][split_idxs]]

def test_shuffle_dataset_split():
    features = np.array([[0,1,1,0],[0,1,1,0],[0,1,1,0]])
    labels = np.array([0,1,1,0])
    dataset = [features,labels]
    dataset = shuffle_dataset_split(dataset)
    if np.all(dataset[0][0]!=dataset[0][1]) or np.all(dataset[0][1]!=dataset[0][2]) or np.all(dataset[0][2]!=dataset[0][0]):
        raise ValueError("Features shuffled incorectly")
    if np.all(np.round(np.mean(features,axis=0))!=labels):
        raise ValueError("Labels shuffled differently then features")


def equalize_nb_dataset_points(ut_split,nonut_split):
    """
    Equilize the size of two points set between two classes
    :return:
    """
    nb_points_ut = len(ut_split[1])
    nb_points_nonut = len(nonut_split[1])
    nb_points = min(nb_points_ut, nb_points_nonut)
    ut_split = [ut_split[0][:nb_points,:],ut_split[1][:nb_points]]
    nonut_split = [nonut_split[0][:nb_points,:],nonut_split[1][:nb_points]]
    features,labels = add_classes_in_split(ut_split,nonut_split)
    return [features,labels]

def resample_nb_dataset_points(ut_split,nonut_split):
    """
    Resample the data points in the given 'ut_split' and 'nonut_split' datasets to ensure both have an equal number of points.
    Add points to the class that is smaller by repeating points from the beggining of the class.
    Parameters:
        ut_split (tuple): A tuple containing feature array and label array for the 'ut' dataset.
        nonut_split (tuple): A tuple containing feature array and label array for the 'nonut' dataset.

    Returns:
        list: A list containing two elements - the resampled features and their corresponding labels.
    """
    nb_points_ut = len(ut_split[1])
    nb_points_nonut = len(nonut_split[1])
    if nb_points_nonut>nb_points_ut:
        ut_split[0] = np.concatenate([ut_split[0],ut_split[0][:nb_points_nonut - nb_points_ut,:]],axis=0)
        ut_split[1] = np.concatenate([ut_split[1], ut_split[1][:nb_points_nonut - nb_points_ut]],axis=0)
    else:
        nonut_split[0] = np.concatenate([nonut_split[0], ut_split[0][:nb_points_ut-nb_points_nonut, :]], axis=0)
        nonut_split[1] = np.concatenate([nonut_split[1], ut_split[1][:nb_points_ut-nb_points_nonut]], axis=0)
    features,labels = add_classes_in_split(ut_split,nonut_split)
    return [features,labels]


def add_classes_in_split(ut_split,nonut_split):
    """
    Combines features and labels from two different splits.

    Parameters:
    ut_split: tuple of numpy.ndarray
        A tuple containing features and labels for one split.
    nonut_split: tuple of numpy.ndarray
        A tuple containing features and labels for another split.

    Returns:
    tuple of numpy.ndarray
        Combined features and labels from both splits.
    """
    features = np.concatenate([ut_split[0], nonut_split[0]], axis=0)
    labels = np.concatenate([ut_split[1],nonut_split[1]])
    return features,labels

if __name__=="__main__":
    test_shuffle_between_epoch()

