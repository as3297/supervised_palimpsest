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
    features = np.reshape(msi_im, newshape=(-1, im_pil_ob.nb_bands)).astype(np.float32)
    row_indices = np.reshape(row_indices, newshape=(-1,)).astype(np.uint32)
    col_indices = np.reshape(col_indices, newshape=(-1,)).astype(np.uint32)
    return features,col_indices,row_indices

def process_chunk(chunk_args):
    """
    :param features_ut: Array-like structure representing the features of user transactions.
    :param features_page: Array-like structure representing the features of the pages.
    :param neighbors: Integer indicating the number of nearest neighbors to find.
    :return: Tuple containing the distances to the nearest neighbors and their corresponding indices.
    """
    features_ut, features_page, xs, ys, n = chunk_args
    # Finding indices of 3 nearest neighbors from features_page to features_ut
    dist = distance.cdist(features_ut, features_page,'euclidean').astype(np.float32)  # Transposed comparison
    n_nn_idx = np.argsort(dist, axis=1)[:, :n]
    dist = np.take_along_axis(dist, n_nn_idx, axis=1)# Picking 3 nearest neighbors
    xs = np.array([xs[n_nn_idx[i,:]] for i in range(len(features_ut))])
    ys = np.array([ys[n_nn_idx[i,:]] for i in range(len(features_ut))])
    return dist,xs,ys,n  # Returning indices of nearest neighbors

# Helper function for processing chunks in parallel

def find_distance_btw_ut_and_folio(data_dir,ut_folio_name, folio_names, class_name,modality,n,nb_processes, chunk_size, save_dir,box=None,):
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
    features_ut = features_ut.astype(np.float32)[:500,:]
    xs_ut = np.array(xs_ut).astype(np.uint32)[:500]
    ys_ut = np.array(ys_ut).astype(np.uint32)[:500]
    print("Done loading undertext features")
    #extract page features
    dict ={}
    for folio_name in folio_names:
        features_page, xs_page, ys_page = load_page(data_dir,folio_name,modality)
        print(f"Done loading page {folio_name} features")
        #increase number of pixel if the page of undertext is the same a page of calculated distances
        same_page = False
        if folio_name == ut_folio_name:
            same_page = True
        dict[folio_name]=find_distance_btw_feat(features_ut, xs_ut, ys_ut, features_page, xs_page, ys_page, n, same_page,nb_processes,chunk_size)
        fpath = os.path.join(save_dir,ut_folio_name+"_"+folio_name+f"_euclid_nn_{n}.pkl")
        save_pickle(fpath,dict)
    return dict

def process_generator(generator):
    """
    :param generator: An iterable that yields tuples of data in the form
        (dist, xs, ys, xs_ut, ys_ut, n, same_page), where:
        - dist: Array of distances
        - xs: Array of x-coordinates for neighbors
        - ys: Array of y-coordinates for neighbors
        - xs_ut: Array of x-coordinates for undertext
        - ys_ut: Array of y-coordinates for undertext
        - n: Number of nearest neighbors to consider
        - same_page: Boolean flag indicating whether to ignore the first
          neighbor if it belongs to the same folio as the undertext

    :return: A dictionary containing:
        - "dist": A list of distances to the n closest neighbors
        - "xs": A list of x-coordinates corresponding to the closest neighbors
        - "ys": A list of y-coordinates corresponding to the closest neighbors
        - "xs_ut": A list of x-coordinates for the undertext elements corresponding to the neighbors
        - "ys_ut": A list of y-coordinates for the undertext elements corresponding to the neighbors
    """
    iter = 0
    dist_acc,xs_acc,ys_acc = [],[],[]
    for dist,xs,ys,n in generator:
        if iter == 0:
            dist_acc = dist
            xs_acc = xs
            ys_acc = ys

        else:
            dist_acc = np.concatenate([dist_acc,dist], axis=1)
            xs_acc = np.concatenate([xs_acc,xs], axis=1)
            ys_acc = np.concatenate([ys_acc,ys], axis=1)

            n_nn_idx = np.argsort(dist_acc, axis=1)[:, 0:n]
            dist_acc = np.take_along_axis(dist_acc, n_nn_idx, axis=1)
            xs_acc = np.take_along_axis(xs_acc, n_nn_idx, axis=1)
            ys_acc = np.take_along_axis(ys_acc, n_nn_idx, axis=1)
        iter += 1
    return dist_acc,xs_acc,ys_acc

def find_distance_btw_feat(features_ut,xs_ut,ys_ut,features_page,xs_page,ys_page,n,same_page,nb_processes,chunk_size):
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
    #if same page as ut page extract more nearest neighbours and then ignore the first neighbour
    if same_page:
        n = n+1

    # Split features_page, xs_page, and ys_page into chunks
    chunks = [(features_ut, features_page[i:min(i + chunk_size, len(features_page))],
               xs_page[i:min(i + chunk_size, len(xs_page))],
               ys_page[i:min(i + chunk_size, len(ys_page))],n)
              for i in range(0, len(features_page), chunk_size)]

    results = Parallel(n_jobs=nb_processes,return_as="generator_unordered")(delayed(process_chunk)(chunk) for chunk in chunks)


    # Collect results from all chunks
    dist,xs,ys = process_generator(results)
    print("Distance calculation complete")
    if same_page:
        dist = dist[:, 1:]
        xs = xs[:, 1:]
        ys = ys[:, 1:]

    return {"dist": dist.tolist(), "xs": xs.tolist(), "ys": ys.tolist(), "xs_ut": xs_ut.tolist(), "ys_ut": ys_ut.tolist()}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Program to extract nearest neighbours location to points of interest")
    parser.add_argument("-r","--root", type=str, default=r"D:", help="Folder where you store all the palimpsests")
    parser.add_argument("-p","--proces", type=int, default=4, help="Number run in parallel")
    parser.add_argument("-ch","--chunk", type=int, default=100, help="Page pixels number for a chunk in distance computation")
    # 3. Parse the arguments
    args = parser.parse_args()
    root_dir = args.root #r"D:" #r"/projects/palimpsests" #
    palimpsest_name = "Verona_msXL"
    nb_processes = args.proces
    chunk_size = args.chunk
    main_data_dir = os.path.join(root_dir, palimpsest_name)
    folio_names = [r"msXL_335v_b",r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b", "msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b", "msXL_334v_b", "msXL_344r_b", "msXL_344v_b", ]
    modality = "M"
    class_name = "undertext"
    n = 3
    box = None

    for folio_ut in folio_names:
        save_dir = os.path.join(main_data_dir,folio_ut, "distances")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        dict = find_distance_btw_ut_and_folio(main_data_dir,folio_ut,folio_names,class_name,
                        modality,n,nb_processes=nb_processes,chunk_size = chunk_size, save_dir=save_dir, box=box)

