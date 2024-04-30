from read_msi_image import conver_pil_msi_ims_to_array
from pil_image_cube import ThumbnailMSI_PIL
import numpy as np
from util import save_json, read_band_list
from util import order_band_list
import os


def create_band_list(cube_dir, txt_save_fpath):
    """
    Create band list
    :param cube_dir:
    :param txt_save_fpath:
    :return:
    """
    bands = [band[:-4] for band in os.listdir(cube_dir) if ".tif" in band]
    bands = [band.split("-")[1] for band in bands]
    ordered_bands = order_band_list(bands)
    with open(txt_save_fpath, "w") as f:
        f.writelines(ordered_bands)

def calculate_max_of_bands(msi_img,bands):
    """Reads image of multiple bands"""
    max_vals = {}
    for idx,band_name in enumerate(bands):
        im = msi_img[idx]
        max_vals[band_name] = find_max(im,np.amax(im))
    return max_vals

def find_max(im,max_val):
    """Find max value without oversaturated pixels
    max_val - scalar, bit depth"""
    if max_val == 1:
        bin_width = 1 / 256.0
    elif max_val == 256:
        bin_width = 1
    else:
        bin_width = 10
    bins = np.arange(0, max_val, bin_width)
    hist, bins = np.histogram(im, bins=bins)
    hist = hist/np.sum(hist)
    for idx in reversed(range(len(hist))):
        if hist[idx] < 0.001 and hist[idx - 1] == 0:
        # print("Hist now {}, hist before {}".format(hist[-idx],hist[-idx-1]))
            max_val = bins[idx - 1]
        else:
            break
    return max_val

def store_max_val(image_dir,band_list_path,fpath_save):
    """
    Read maximum values from image cube of folio "msXL_315r_b"
    :param image_dir:
    :param band_list_path:
    :param fpath_save:
    :return:
    """
    folio_name = "msXL_315r_b"
    bands = read_band_list(band_list_path)
    obj = ThumbnailMSI_PIL(image_dir,folio_name,bands,0,scale_ratio=20)
    nb_bands = len(obj.band_list)
    msi_ims = conver_pil_msi_ims_to_array(obj.msi_img_thumbnail,obj.width,obj.height,nb_bands)
    max_vals = calculate_max_of_bands(msi_ims , bands)
    save_json(fpath_save,max_vals)
    obj.close_all_images()

def store_dataset_split():


if __name__ == "__main__":

