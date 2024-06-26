from read_msi_data_as_array import conver_pil_msi_ims_to_array,FragmentfromMSI_PIL
from pil_image_cube import ThumbnailMSI_PIL,ImageCubePILobject
import numpy as np
from util import save_json, read_band_list, read_json, order_band_list
import os
from read_pixel_coord import points_coord_in_bbox
import csv
from PIL import Image
from skimage import io,transform
from dataset import read_bboxs

def save_the_subset_fragment_band_14():
    osp = os.path.join
    main_path = r"C:\Data\PhD\palimpsest\Victor_data"
    folio_name = r"msXL_319r_b"
    subset = "val"
    band_list_path = osp(main_path, "band_list.txt")
    bands = read_band_list(band_list_path)
    image_dir_path = osp(main_path, folio_name)
    bbox_fpath = osp(main_path, folio_name, "dataset_split.json")
    bbox_dict = read_json(bbox_fpath)
    bbox = read_bboxs(subset, bbox_dict)
    band_idx =14
    folio_ob = ImageCubePILobject(image_dir_path, folio_name, bands, 0)
    fragment_im = FragmentfromMSI_PIL(folio_ob, 0,bbox).ims_img[band_idx]
    fragment_im= (fragment_im*255).astype(np.uint8)
    save_path = osp(image_dir_path,"miscellaneous",subset +"_"+bands[band_idx]+".png")
    io.imsave(save_path,fragment_im)
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
    def store_coord_in_dict(cd):
        return {"x1":cd[0],"y1":cd[1],"x2":cd[2],"y2":cd[3]}
    d = {}
    d["test_bbox"] = store_coord_in_dict([60,1197, 4680, 2288])
    d["train_bbox"] = store_coord_in_dict([60, 2288, 4680, 5857])
    d["val_bbox"] = store_coord_in_dict([60, 5857, 4680, 6441])
    save_json(r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\dataset_split.json",d)

def calc_points_in_each_datasplit(class_name,class_mask_fpath):
    d_bbox = read_json(r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\dataset_split.json")
    csvfile = open(r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\nb_points_per_split_{}.csv".format(class_name),"w",newline='')
    csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for split_name,bbox in d_bbox.items():
        bbox = [bbox["x1"],bbox["y1"],bbox["x2"],bbox["y2"]]
        _, _, nb_coords = points_coord_in_bbox(class_mask_fpath, bbox)
        csvwriter.writerow([split_name.split("_")[0],nb_coords])
    csvfile.close()

def calc_points_in_each_class(d_classes):
    """
    Stores csv file with number of points for each datasplit in each class
    :param d_classes: {class_name1:class_mask_fpath1,...}
    :return: None
    """
    for class_name,fpath in d_classes.items():
        calc_points_in_each_datasplit(class_name,fpath)

def rotate_imgs(im_dir,save_dir):
    for f in os.listdir(im_dir):
        if ".tif" in f:
            im = Image.open(os.path.join(im_dir,f))
            rotation = Image.ROTATE_270
            im = im.transpose(rotation)
            im.save(os.path.join(save_dir,f))


if __name__ == "__main__":
    #rotate_imgs(r"c:\Data\PhD\palimpsest\Victor_data\msXL_319r_b",r"C:\Data\PhD\palimpsest\Victor_data\msXL_319r_b\rotated")
    save_the_subset_fragment_band_14()

