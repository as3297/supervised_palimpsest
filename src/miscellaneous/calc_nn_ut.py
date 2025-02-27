import sys
import os
# Add the root directory (one level up) to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from read_data import read_msi_image_object, read_subset_features
from msi_data_as_array import FullImageFromPILImageCube
from util import save_pickle
from scipy.spatial import distance
from joblib import Parallel, delayed
import numpy as np
import argparse

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
    row_indices = np.reshape(row_indices, newshape=(-1,)).astype(np.uint32)
    col_indices = np.reshape(col_indices, newshape=(-1,)).astype(np.uint32)
    return features,col_indices,row_indices

def find_distance_btw_ut_and_folio_frag(features_ut,features_page,xs,ys, neighbors=3):
    """
    :param features_ut: Array-like structure representing the features of user transactions.
    :param features_page: Array-like structure representing the features of the pages.
    :param neighbors: Integer indicating the number of nearest neighbors to find.
    :return: Tuple containing the distances to the nearest neighbors and their corresponding indices.
    """
    # Finding indices of 3 nearest neighbors from features_page to features_ut
    dist = distance.cdist(features_ut, features_page,'euclidean').astype(np.float32)  # Transposed comparison
    idx_dist_sorted_page_to_ut = np.argsort(dist, axis=1)
    n_nn_idx = idx_dist_sorted_page_to_ut[:, :neighbors]
    dist = np.take_along_axis(dist, n_nn_idx, axis=1)# Picking 3 nearest neighbors
    xs = np.repeat(xs[np.newaxis,:], repeats=len(features_ut), axis=0)
    ys = np.repeat(ys[np.newaxis,:], repeats=len(features_ut), axis=0)
    xs = np.take_along_axis(xs, n_nn_idx, axis=1)
    ys = np.take_along_axis(ys, n_nn_idx, axis=1)
    return dist,xs,ys  # Returning indices of nearest neighbors

# Helper function for processing chunks in parallel
def process_chunk(chunk_args):
    features_ut, features_page_chunk, xs_page_chunk, ys_page_chunk, n = chunk_args
    return find_distance_btw_ut_and_folio_frag(features_ut, features_page_chunk, xs_page_chunk, ys_page_chunk, n)

def find_distance_btw_ut_and_folio(data_dir,ut_folio_name, folio_name, class_name,modality,n_org,nb_processes, box=None,):
    """
    :param data_dir: Directory path where the data is stored.
    :param ut_folio_name: Name of the undertext folio to process.
    :param folio_name: Name of the folio or list of folio names for which distances are calculated.
    :param class_name: The class name used to filter features when reading data.
    :param modality: Specific modality of features (e.g., grayscale, RGB) used for processing.
    :param n_org: Original number of neighboring pixels for distance computation.
    :param nb_processes: Number of processes to use for parallel computation.
    :param box: Optional bounding box to limit feature extraction to a specific region.
    :return: Dictionary containing distances between undertext features and page features for each folio, along with associated data.
    """
    #extract ut features
    features_ut,xs_ut,ys_ut = read_subset_features(data_dir,ut_folio_name,class_name,modality,box)
    features_ut = features_ut.astype(np.float32)
    xs_ut = np.array(xs_ut).astype(np.uint32)
    ys_ut = np.array(ys_ut).astype(np.uint32)
    print("Done loading undertext features")
    #extract page features
    dict ={"xs_ut":xs_ut.tolist(),"ys_ut":ys_ut.tolist()}
    for folio_name in folio_names:
        features_page, xs_page, ys_page = load_page(data_dir,folio_name,modality)
        features_page = features_page.astype(np.float32)
        xs_page = xs_page
        ys_page = ys_page
        print(f"Done loading page {folio_name} features")
        #increase number of pixel if the page of undertext is the same a page of calculated distances
        if folio_name == ut_folio_name:
            n = n_org+1
        else:
            n = n_org
        same_page = False
        if folio_name == ut_folio_name:
            same_page = True
        dict[folio_name]=find_distance_btw_feat(features_ut, xs_ut, ys_ut, features_page, xs_page, ys_page, n, same_page,nb_processes)
    return dict

def find_distance_btw_feat(features_ut,xs_ut,ys_ut,features_page,xs_page,ys_page,n,same_page,nb_processes):
    """
    :param features_ut: Feature set from the under-text section.
    :param xs_ut: X-coordinates associated with the under-text features.
    :param ys_ut: Y-coordinates associated with the under-text features.
    :param features_page: Feature set from the reference page.
    :param xs_page: X-coordinates associated with the reference page features.
    :param ys_page: Y-coordinates associated with the reference page features.
    :param n: Number of nearest neighbors to retrieve.
    :param same_page: Boolean flag indicating whether the under-text and reference features are from the same page.
    :return: Dictionary containing the distances, and corresponding coordinates of the nearest neighbors.
    """
    #process image chunk by chunk to save memory
    chunk_size = 100  # Set a reasonable chunk size

    # Split features_page, xs_page, and ys_page into chunks
    chunks = [(features_ut, features_page[i:min(i + chunk_size, len(features_page))],
               xs_page[i:min(i + chunk_size, len(xs_page))],
               ys_page[i:min(i + chunk_size, len(ys_page))], n)
              for i in range(0, len(features_page), chunk_size)]

    results = Parallel(n_jobs=nb_processes, batch_size=2,max_nbytes=None, prefer="processes",backend="loky")(delayed(process_chunk)(chunk) for chunk in chunks)

    print("Distance calculation complete")
    # Collect results from all chunks

    dist, xs, ys = [], [], []
    for dist_chunk, xs_chunk, ys_chunk in results:
        dist.append(dist_chunk)
        xs.append(xs_chunk)
        ys.append(ys_chunk)

    dist = np.concatenate(dist, axis=1)
    xs = np.concatenate(xs, axis=1)
    ys = np.concatenate(ys, axis=1)
    idx_dist_sorted = np.argsort(dist, axis=1)
    #ignore first neighbour if it is the same folio as undertext's folio
    if same_page:
        n_nn_idx = idx_dist_sorted[:, 1:n]
    else:
        n_nn_idx = idx_dist_sorted[:, 0:n]

    dist = np.take_along_axis(dist, n_nn_idx, axis=1)
    xs = np.take_along_axis(xs, n_nn_idx, axis=1)
    ys = np.take_along_axis(ys, n_nn_idx, axis=1)
    dist_dict = {"dist":dist.tolist(),"xs":xs.tolist(),"ys":ys.tolist()}
    return dist_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Program to extract nearest neighbours location to points of interest")
    parser.add_argument("--root", type=str, default=r"D:", help="Folder where you store all the palimpsests")
    parser.add_argument("--proces", type=int, default=4, help="Number run in parallel")
    # 3. Parse the arguments
    args = parser.parse_args()
    root_dir = args.root #r"D:" #r"/projects/palimpsests" #
    palimpsest_name = "Verona_msXL"
    nb_processes = args.proces
    main_data_dir = os.path.join(root_dir, palimpsest_name)
    folio_names = [r"msXL_335v_b",]#r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b", "msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b", "msXL_334v_b", "msXL_344r_b", "msXL_344v_b", ]
    modality = "M"
    class_name = "undertext"
    n = 3

    box = None
    for folio_ut in folio_names:
        dict = find_distance_btw_ut_and_folio(main_data_dir,folio_ut,folio_names,class_name,modality,n,nb_processes=nb_processes,box=box)
        fpath = os.path.join(main_data_dir,folio_ut,f"euclid_nn_{n}.pkl")
        save_pickle(fpath,dict)
