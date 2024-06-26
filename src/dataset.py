import numpy as np
from read_pixel_coord import points_coord_in_bbox
from read_msi_data_as_array import PointsfromMSI_PIL,FullImageFromPILImageCube
from pil_image_cube import ImageCubePILobject
from util import read_max_vals,read_band_list,read_json,extend_json,debugging_dict
from copy import deepcopy
import os
from matplotlib import pyplot as plt
osp = os.path.join

def test_load_data_for_training_from_folio(main_path, folio_name, bands, bbox_dict):
    dataset = load_data_for_training_from_folio(main_path, folio_name, bands, bbox_dict)
    for key,subset in dataset.items():
        image_dir_path = osp(main_path, folio_name)
        folio_ob = ImageCubePILobject(image_dir_path, folio_name, bands, 0)
        full_im = FullImageFromPILImageCube(folio_ob,0).ims_img
        full_im_copy = deepcopy(full_im)
        full_im_white = np.ones_like(full_im_copy)
        coords = subset[2]
        features = subset[0].transpose([1,0])
        print("Nb of points ", len(subset[1]))
        coords_x = np.array(list(zip(*coords))[0])
        coords_y = np.array(list(zip(*coords))[1])
        res = features-full_im_copy[:,coords_y,coords_x]
        if np.mean(res)>0:
            raise ValueError("The extracted values do not match values on the image")
        #full_im_white[:, coords_y, coords_x] = 0
        #for i in range(features.shape[0]):
        #    plt.figure(folio_name+key+"_real_im"+"band_"+str(i))
        #    plt.imshow(full_im[i],cmap="gray")
        #    plt.figure(folio_name+key + "_restored_im" + "band_" + str(i))
        #    plt.imshow(full_im_copy[i],cmap="gray")
        #    plt.figure(folio_name + key + "_b&w_im" + "band_" + str(i))
        #    plt.imshow(full_im_white[i], cmap="gray")
        #plt.show()


def load_data_for_training(model_path: str, modalities: list, debugging=False):
    """
    Load features and labels from multiple folios
    """
    main_path = r"C:\Data\PhD\palimpsest\Victor_data"
    folios = [r"msXL_315r_b",r"msXL_319r_b"]
    band_list_path = osp(main_path, "band_list.txt")
    bands = read_band_list(band_list_path)
    bands = sublist_of_bands(bands, modalities)
    bbox_dicts = []
    dataset = {}
    for idx, folio_name in enumerate(folios):
        print("Folio name ",folio_name)
        bbox_fpath = osp(main_path, folio_name, "dataset_split.json")
        bbox_dict = read_json(bbox_fpath)
        if debugging:
            bbox_dict = debugging_dict
        bbox_dicts.append(bbox_dict)

        folio_dataset = load_data_for_training_from_folio(main_path, folio_name, bands, bbox_dict)

        test_load_data_for_training_from_folio(main_path, folio_name, bands, bbox_dict)
        if idx == 0:
            for subset, subset_val in folio_dataset.items():
                dataset[subset] = [subset_val[0], subset_val[1]]
        else:
            for subset, subset_val in folio_dataset.items():
                dataset[subset][0] = np.concatenate([dataset[subset][0], subset_val[0]], axis=0)
                dataset[subset][1] = np.concatenate([dataset[subset][1], subset_val[1]], axis=0)
        print("Folio {} number of points in train ut set={}".format(folio_name,len(dataset["train_ut"][1])))
        if len(model_path)>0:
            save_data_parameters(model_path, modalities, bbox_dicts, folios)
    return dataset["train_ut"],dataset["train_nonut"],dataset["val_ut"],dataset["val_nonut"]


def save_data_parameters(save_path, modalities: list, bbox_dicts: list, folios: list):
    d = {}
    d["folios"] = folios
    d["modalities"] = modalities
    d["coord_boxs"] = bbox_dicts
    extend_json(osp(save_path, "dataset_par.json"), d)


def load_data_for_training_from_folio(main_path, folio_name, bands, bbox_dict):
    """
    Load features and labels from one folio
    :return:
    [],[],[],[] - first list train data [features,labels] for undertext,
              second list train data [features,labels] for nonundertext,
              third list validation data [features,labels] for  undertext,
              forth list validation data [features,labels] for  nonundertext

    """
    max_val_path = osp(main_path, "bands_max_val.json")
    max_vals = read_max_vals(max_val_path, bands)
    image_dir_path = osp(main_path, folio_name)

    ut_mask_path = osp(image_dir_path, "mask", folio_name + r"-undertext_black.png")
    nonut_mask_path = osp(image_dir_path, "mask", folio_name + r"-not_undertext_black.png")
    im_msi_pil_ob = ImageCubePILobject(image_dir_path, folio_name, bands, 0)

    trainset_ut, trainset_nonut = read_point_class_ut_nonut_split(bbox_dict, im_msi_pil_ob,
                                                                  ut_mask_path, nonut_mask_path, max_vals, "train")
    valset_ut, valset_nonut = read_point_class_ut_nonut_split(bbox_dict, im_msi_pil_ob,
                                                              ut_mask_path, nonut_mask_path, max_vals, "val")

    return {"train_ut": trainset_ut, "train_nonut": trainset_nonut, "val_ut": valset_ut, "val_nonut": valset_nonut}


def shuffle_between_epoch(set_ut,set_nonut):
  set_ut = shuffle_dataset_split(set_ut)
  set_nonut = shuffle_dataset_split(set_nonut)
  dataset = equalize_nb_dataset_points(set_ut, set_nonut)
  dataset = shuffle_dataset_split(dataset)
  nb_samples = len(dataset[1])
  return dataset,nb_samples
def read_points_class(fpath_image_mask,im_msi_pil_ob,bbox,label,max_vals):
    xs, ys, nb_coords = points_coord_in_bbox(fpath_image_mask, bbox)
    labels = np.array([label]*nb_coords)
    coords = list(zip(xs,ys))
    points_object = PointsfromMSI_PIL(im_msi_pil_ob, max_vals, coords)
    features = points_object.points
    return features,labels,coords

def sublist_of_bands(bands,modalities=["M"]):
  bands_subset = []
  for band in bands:
      for modality in modalities:
        if modality in band:
          bands_subset.append(band)
  return bands_subset

def read_bboxs(split_name,d_bboxs):
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

def read_point_class_ut_nonut_split(bbox_dict,im_msi_pil_ob,ut_mask_path,nonut_mask_path,max_vals,split_name):
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
    bbox = read_bboxs(split_name, bbox_dict)
    feat_ut, labels_ut,coords_ut = read_points_class(ut_mask_path, im_msi_pil_ob, bbox, 1, max_vals)
    feat_nout, labels_nonut,coords_nonut = read_points_class(nonut_mask_path, im_msi_pil_ob, bbox, 0, max_vals)
    return [feat_ut,labels_ut,coords_ut],[feat_nout,labels_nonut,coords_nonut]

def shuffle_dataset_split(data_subset,show_split_idxs=False):
    """Shuffle dataset subset points"""
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


def add_classes_in_split(ut_split,nonut_split):
    features = np.concatenate([ut_split[0], nonut_split[0]], axis=0)
    labels = np.concatenate([ut_split[1],nonut_split[1]])
    return features,labels

if __name__=="__main__":
    load_data_for_training("","M",True)

