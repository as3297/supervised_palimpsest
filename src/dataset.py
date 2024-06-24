import numpy as np
from read_pixel_coord import points_coord_in_bbox
from read_msi_data_as_array import PointsfromMSI_PIL

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
    points_object = PointsfromMSI_PIL(im_msi_pil_ob, max_vals, list(zip(xs,ys)))
    features = points_object.points
    return features,labels

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

def read_features_labels(bbox_dict,im_msi_pil_ob,ut_mask_path,nonut_mask_path,max_vals,split_name):
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
    feat_ut, labels_ut = read_points_class(ut_mask_path, im_msi_pil_ob, bbox, 1, max_vals)
    feat_nout, labels_nonut = read_points_class(nonut_mask_path, im_msi_pil_ob, bbox, 0, max_vals)
    return [feat_ut,labels_ut],[feat_nout,labels_nonut]

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
    test_shuffle_classes_in_split()

