import numpy as np
from src.read_data import read_msi_image_object, read_rectengular_patch_coord
from src.msi_data_as_array import FullImageFromPILImageCube
from src.read_data import read_subset_features
import os
import time
from concurrent.futures import ThreadPoolExecutor
from skimage import io
from src.util import read_json


def calc_cov_matrix(features):

    choices = np.random.randint(0, len(features), 1000)
    features = np.array([features[i] for i in choices])
    cov = np.cov(features.T)
    return cov

def calc_aver(features):
    aver = np.mean(features,0)
    return aver

def Mahalanobis_dist(x,cov_inv,aver):
    """
    Calculates the Mahalanobis distance between a point and a distribution defined by its covariance matrix and mean.

    Args:
        x (np.ndarray): The data point as a vector (1D array).
        cov (np.ndarray): Covariance matrix of the distribution.
        aver (np.ndarray): Mean vector of the distribution.

    Returns:
        float: The Mahalanobis distance.
    """
    diff = x - aver
    dist = np.sqrt(np.dot(np.dot(diff.T, cov_inv),diff))
    return dist

def calculate_distance(i, j, point, mean, cov_inv):
    return i, j, Mahalanobis_dist(point, cov_inv, mean)

def calc_Mahalanobis_dist_im(x, features):
    """
    Credit to bjd2385/rx. https://github.com/bjd2385/rx/blob/master/mahalanobis.py
    """
    cov_matrix = calc_cov_matrix(features)
    cov_inv = np.linalg.inv(cov_matrix)
    aver = calc_aver(features)
    rows,cols = x.shape[0],x.shape[1]
    distances = np.zeros([rows, cols])

    num_cores = os.cpu_count()
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        future_to_index = {
            executor.submit(calculate_distance, i, j, x[i][j], aver, cov_inv): (i, j)
            for i in range(rows) for j in range(cols)
        }

        for future in future_to_index:
            i, j, distance = future.result()
            distances[i][j] = distance
    return distances

def test_read_page_coord():
    main_dir = r"c:\Data\PhD\palimpsest\Victor_data"
    palimpsest_name = "Paris_Coislin"
    folio_name = r"Par_coislin_393_054r"
    row_start, row_end, col_start, col_end = read_rectengular_patch_coord(main_dir, palimpsest_name, folio_name)
    page_mask_page = os.path.join(main_dir, palimpsest_name, folio_name, "mask", f"{folio_name}-page_black.png")
    page_mask = io.imread(page_mask_page, as_gray=True)
    page_mask[page_mask>=0.5]=1
    page_mask[page_mask<0.5]=0
    art_page_mask = np.ones_like(page_mask)
    art_page_mask[row_start:row_end,col_start:col_end] = 0
    diff = page_mask-art_page_mask
    if np.sum(diff)>0:
        raise Exception("Image and artificial mask do not match")

def read_overtext_mask(main_dir,palimpsest_name,folio_name):
    overtext_mask = os.path.join(main_dir, palimpsest_name, folio_name, "mask", f"{folio_name}-overtext_black.png")
    overtext_mask = io.imread(overtext_mask, as_gray=True)
    if np.max(overtext_mask)>1:
        overtext_mask = overtext_mask/np.max(overtext_mask)
    overtext_mask[overtext_mask>=0.5]=1
    overtext_mask[overtext_mask<0.5]=0
    return overtext_mask

def thresh_feature_values(main_dir,palimpsest_name,features,im_pil_obj):
    dict_thresh = read_json(os.path.join(main_dir,palimpsest_name,"bg_1_upper_thresh.json"))
    dict_means_bg = read_json(os.path.join(main_dir,palimpsest_name,"bg_1_means.json"))
    band_list = im_pil_obj.band_list
    thresh_values = np.zeros((im_pil_obj.nb_bands,))
    for i,band_name in enumerate(band_list):
        thresh_values[i] = dict_thresh[band_name]
    ndim = features.ndim
    if ndim == 3:
        for i,band_name in enumerate(band_list):
            features[:,:,i][features[:,:,i] > thresh_values[i]] = dict_means_bg[band_name]
    else:
        for i, band_name in enumerate(band_list):
            features[:,i][features[:,i] > thresh_values[i]] = dict_means_bg[band_name]
    features = np.clip(features,a_min=0,a_max=thresh_values)
    return features

def apply_RXD(main_dir,palimpsest_name,folio_name,patch_name,modality):
    palimpsest_dir = os.path.join(main_dir,palimpsest_name)
    im_pil_ob = read_msi_image_object(palimpsest_dir, folio_name, modality)
    msi_im_obj = FullImageFromPILImageCube(pil_msi_obj=im_pil_ob)
    msi_im = msi_im_obj.ims_img
    msi_im = np.transpose(msi_im,axes=[1,2,0])
    col_start,row_start,col_end,row_end = read_rectengular_patch_coord(main_dir, palimpsest_name, folio_name,patch_name)
    dist_full_page = np.zeros(([msi_im.shape[0],msi_im.shape[1]]))
    msi_im = msi_im[row_start:row_end,col_start:col_end,:]
    msi_im = thresh_feature_values(main_dir,palimpsest_name,msi_im,im_pil_ob)
    print("Shape msi_im",msi_im.shape)
    class_name = "bg_1"
    features_bg,_,_ = read_subset_features(palimpsest_dir, folio_name, class_name, modality, None)
    features_bg = thresh_feature_values(main_dir,palimpsest_name,features_bg,im_pil_ob)
    start_time = time.time()
    dist = calc_Mahalanobis_dist_im(msi_im, features_bg)
    nb_pixels = msi_im.shape[0]*msi_im.shape[1]
    time_period = time.time()-start_time
    print(f"Calculating Mahalonobis distance over {nb_pixels} pixels, takes {time_period} seconds")
    dist_full_page[row_start:row_end,col_start:col_end] = dist
    overtext_mask = read_overtext_mask(main_dir, palimpsest_name, folio_name)
    dist = dist_full_page
    #dist[overtext_mask==0] = 0
    dist = dist/np.amax(dist)
    dist = np.clip(dist, 0, 1)
    dist = (255 * dist).astype(np.uint8)
    return dist

if __name__=="__main__":
    main_dir = r"c:\Data\PhD\palimpsest\Victor_data"
    palimpsest_name = "Paris_Coislin"
    folio_name = r"Par_coislin_393_054r"
    modality = "M"
    exp_name = "mah_dis_with_ot_varnish_to_mean_bg_scale_new"
    patch_name = "row_rxd_test"
    dist = apply_RXD(main_dir, palimpsest_name, folio_name, patch_name,modality)
    io.imsave(os.path.join(main_dir,palimpsest_name,folio_name,"miscellaneous",f"{folio_name}_{patch_name}_{exp_name}.png"),dist)