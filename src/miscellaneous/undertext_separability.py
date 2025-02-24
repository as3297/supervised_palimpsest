import sys
import os
# Add the root directory (one level up) to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from read_data import read_msi_image_object, read_subset_features
from msi_data_as_array import FullImageFromPILImageCube
from util import save_json
from scipy.spatial import distance
import numpy as np

def bhattacharyya_coefficient(hist1, hist2):
  """
  Calculates the Bhattacharyya coefficient between two histograms.
  Args:
    hist1: First histogram (NumPy array).
    hist2: Second histogram (NumPy array).

  Returns:
    The Bhattacharyya coefficient (float).
  """

  # Ensure the histograms have the same size
  if len(hist1) != len(hist2):
    raise ValueError("Histograms must have the same size")

  # Calculate the Bhattacharyya coefficient
  bc = np.sum(np.sqrt(np.multiply(hist1, hist2)))

  return bc

def load_page(data_dir,folio_name,modality):
    """
    :param data_dir: Directory path where the data files are located
    :param folio_name: Name of the image folio to be loaded
    :param modality: Specific modality or type of the image data to process
    :return: A tuple containing the features matrix, row indices, and column indices
    """
    im_pil_ob = read_msi_image_object(data_dir, folio_name, modality)
    msi_im_obj = FullImageFromPILImageCube(pil_msi_obj=im_pil_ob)
    im_shape = (im_pil_ob.height, im_pil_ob.width)
    msi_im = msi_im_obj.ims_img
    row_indices, col_indices = np.meshgrid(np.arange(im_shape[0]), np.arange(im_shape[1]), indexing='ij')
    features = np.reshape(msi_im, newshape=(-1, im_pil_ob.nb_bands))
    row_indices = np.reshape(row_indices, newshape=(-1,))
    col_indices = np.reshape(col_indices, newshape=(-1,))
    return features,col_indices,row_indices

def find_distance_btw_ut_and_folio_frag(features_ut,features_page,xs,ys, neighbors=3):
    """
    :param features_ut: Array-like structure representing the features of user transactions.
    :param features_page: Array-like structure representing the features of the pages.
    :param neighbors: Integer indicating the number of nearest neighbors to find.
    :return: Tuple containing the distances to the nearest neighbors and their corresponding indices.
    """
    # Finding indices of 3 nearest neighbors from features_page to features_ut
    dist= distance.cdist(features_ut, features_page,'euclidean')  # Transposed comparison
    idx_dist_sorted_page_to_ut = np.argsort(dist, axis=1)
    n_nn_idx = idx_dist_sorted_page_to_ut[:, :neighbors]
    dist = np.take_along_axis(dist, n_nn_idx, axis=1)# Picking 3 nearest neighbors
    xs = np.repeat(xs[np.newaxis,:], repeats=len(features_ut), axis=0)
    ys = np.repeat(ys[np.newaxis,:], repeats=len(features_ut), axis=0)
    xs = np.take_along_axis(xs, n_nn_idx, axis=1)
    ys = np.take_along_axis(ys, n_nn_idx, axis=1)
    return dist,xs,ys  # Returning indices of nearest neighbors

def find_distance_btw_ut_and_folio(data_dir,ut_folio_name, folio_name, class_name,modality,n, box=None,):
    """
    :param data_dir: Directory path where the necessary data files are stored.
    :param ut_folio_name: Name of the undertext folio for which features are being processed.
    :param folio_name: Name of the folio whose features are compared against the undertext folio.
    :param class_name: Class name used for categorizing or filtering data during feature loading.
    :param modality: Type or mode of the data, which determines how features are loaded (e.g., image, text).
    :param n: Number of nearest neighbors to find during distance calculation.
    :param box: Optional parameter specifying a bounding box to limit feature processing (default is None).
    :return: A dictionary containing distance metrics and coordinates ('dist', 'xs_ut', 'ys_ut', 'xs', 'ys') between features of the undertext folio and the specified folio.
    """
    #extract ut features
    features_ut,xs_ut,ys_ut = read_subset_features(data_dir,ut_folio_name,class_name,modality,box)
    features_ut = features_ut.astype(np.float32)
    xs_ut = xs_ut
    ys_ut = ys_ut
    print("Done loading undertext features")
    dist_dict = {}
    #extract page features
    features_page, xs_page, ys_page = load_page(data_dir,folio_name,modality)
    features_page = features_page.astype(np.float32)
    print(f"Done loading page {folio_name} features")
    #process image chunk by chunk to save memory
    chunk_size = 10000  # Set a reasonable chunk size
    dist,xs,ys = [],[],[]
    #increase number of pixel if the page of undertext is the same a page of calculated distances
    if folio_name == ut_folio_name:
        n = n+1
    for i in range(0, len(features_page), chunk_size):
        last_idx = min(i + chunk_size,len(features_page))
        dist_chunk, xs_chunk, ys_chunk = find_distance_btw_ut_and_folio_frag(features_ut, features_page[i:last_idx],xs_page[i:last_idx],ys_page[i:last_idx], n)
        dist.append(dist_chunk)
        xs.append(xs_chunk)
        ys.append(ys_chunk)
        print(f"Percentage of {len(features_page)/(i+len(features_page))} pixels left")

    dist = np.concatenate(dist, axis=1)
    xs = np.concatenate(xs, axis=1)
    ys = np.concatenate(ys, axis=1)
    idx_dist_sorted = np.argsort(dist, axis=1)
    #ignore first neighbour if it is the same folio as undertext's folio
    if folio_name == ut_folio_name:
        n_nn_idx = idx_dist_sorted[:, 1:n]
    else:
        n_nn_idx = idx_dist_sorted[:, 0:n]

    dist = np.take_along_axis(dist, n_nn_idx, axis=1)
    xs = np.take_along_axis(xs, n_nn_idx, axis=1)
    ys = np.take_along_axis(ys, n_nn_idx, axis=1)
    dist_dict[folio_name] = {"dist":dist,"xs_ut":xs_ut,"ys_ut":ys_ut,"xs":xs,"ys":ys}
    return dist_dict


#find distance a Euclidean between every undertext feature and every pixel of every palimpsest
#find the coordinates of tree pixels with the smallest distance
#create a map using this pixels
#color every pixel from this map according to index of

if __name__ == "__main__":
    
    root_dir = r"/projects/palimpsests" #r"D:"
    palimpsest_name = "Verona_msXL"
    main_data_dir = os.path.join(root_dir, palimpsest_name)
    folio_names = ["msXL_335v_b", r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
                   "msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b",
                   "msXL_334v_b", "msXL_344r_b", "msXL_344v_b", ]
    modality = "M"
    class_name = "undertext"
    n = 3
    box = None
    dict_list = []
    for folio_ut in folio_names:
        for folio_name in folio_names:
            dict = find_distance_btw_ut_and_folio(main_data_dir,folio_names[0],folio_name,class_name,modality,n,box=box,)
            dict_list.append(dict_list)
        dict_folios = {folio_names[0]:dict_list}
        fpath = os.path.join(main_data_dir,folio_names[0],f"euclid_nn_{n}.json")
        save_json(fpath,dict_folios)
