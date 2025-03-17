from pil_image_cube import ImageCubePILobject
from msi_data_as_array import PointsfromMSI_PIL,FragmentfromMSI_PIL
from pixel_coord import points_coord_in_bbox
from util import read_json, read_band_list, read_split_box_coord
import numpy as np
import skimage.io as io
import os

osp = os.path.join


def read_msi_image_patch(main_directory, folio, modality,box):
    """
    Read msi image patch
    :param main_directory: Directory path where the MSI images are stored.
    :param folio: Identifier or name used to select the specific MSI image folder.
    :param modality: Imaging modality type to filter or locate the desired MSI image.
    :param box: Bounding box coordinates used to extract a specific fragment from the image.[left, upper, right, and lower]
    :return: Processed MSI image fragment as a result of the box extraction.
    """
    im_msi_pil = read_msi_image_object(main_directory, folio, modality)
    im_msi = FragmentfromMSI_PIL(im_msi_pil,box).ims_img
    im_msi = np.reshape(im_msi, (-1, im_msi.shape[-1]))

    # Generate the two lists of corresponding coordinates for x and y inside the box
    x_coords = range(box[0], box[2])
    y_coords = range(box[1], box[3])
    # Generate the meshgrid of coordinates for pixels inside the bounding box
    x_coords, y_coords = np.meshgrid(x_coords, y_coords)
    return im_msi,x_coords.flatten().tolist(),y_coords.flatten().tolist()

def read_msi_image_object(main_directory, folio, modality):
    """
    Reads subset features from a given directory for a specified folio and class name.

    Parameters:
    main_directory (str): The main directory path containing the necessary data files.
    folio (str): The name of the folio to read.
    modality (str): The modality to be used for reading the band list.

    Returns:
    ImageCubePILobject: An object containing the image data and associated bands.
    """
    band_list_file = osp(main_directory, "band_list.txt")
    band_list = read_band_list(band_list_file, modality)
    rotate_angle = 0  # Assigning a default rotate angle
    image_cube_pil_object = ImageCubePILobject(main_directory, folio, band_list, rotate_angle)
    return image_cube_pil_object

def read_ot_mask(main_dir, palimpsest_name, folio_name,box):
    ot_path = os.path.join(main_dir, palimpsest_name, folio_name, "mask", f"{folio_name}-overtext_black.png")
    ot_im = io.imread(ot_path, as_gray=True)
    if np.max(ot_im) > 1:
        ot_im = ot_im / np.max(ot_im)
    if not box is None:
        bbox_fpath = os.path.join(main_dir, palimpsest_name, folio_name, "dataset_split.json")
        bbox_dict = read_json(bbox_fpath)
        bbox = read_split_box_coord(box, bbox_dict)
        ot_im = ot_im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return ot_im




def read_x_y_coords(main_dir,folio_name,class_name,im_pil_ob,box=None):
    """
    Reads X and Y coordinates for a given folio name, class, and image object based on a bounding box.

    Arguments:
     main_dir (str): The main directory path.
     folio_name (str): The name of the folio.
     class_name (str): The class name corresponding to the image.
     im_pil_ob: Image object used to get the dimensions of the image.
     box (str)(optional): A bounding box to define an area within the image. Defaults to the entire image if not provided.

    Returns:
     xs: List of X coordinates within the bounding box.
     ys: List of Y coordinates within the bounding box.

    If a bounding box is provided, the function will read and utilize it to determine the coordinates. If not, it will use the entire image dimensions.
    """
    if not box is None:
        if type(box) is str:
            bbox_fpath = osp(main_dir, folio_name, "dataset_split.json")
            bbox_dict = read_json(bbox_fpath)
            bbox = read_split_box_coord(box, bbox_dict)
        else:
            bbox = box
    else:
        bbox = [0,0,im_pil_ob.width-1,im_pil_ob.height-1]

    fpath_image_mask = os.path.join(main_dir, folio_name, "mask", f"{folio_name}-{class_name}_black.png")
    xs, ys, _ = points_coord_in_bbox(fpath_image_mask, bbox)
    return xs,ys

def read_subset_features(main_dir,folio_name,class_name,modality,box=None):
    """
    Reads subset of features from a given MSI (Mass Spectrometry Imaging) image dataset.

    Parameters:
    main_dir: str
        The main directory path containing the dataset.
    folio_name: str
        The name of the folio or folder containing specific image files.
    class_name: str
        The name of the classification or label associated with the data.
    modality: str
        The imaging modality name (e.g., intensity or ion mode).
    box: Optional[tuple], default=None
        A bounding box defined by top-left and bottom-right coordinates to restrict the area of interest. If None, the entire region is considered.

    Returns:
    tuple
        A tuple containing:
            - features: list
                Extracted features from the specified coordinates in the image.
            - xs: list
                X-coordinates of the extracted points.
            - ys: list
                Y-coordinates of the extracted points.
    """
    im_pil_ob = read_msi_image_object(main_dir,folio_name,modality)
    xs,ys = read_x_y_coords(main_dir,folio_name,class_name,im_pil_ob,box)
    points_object = PointsfromMSI_PIL(pil_msi_obj=im_pil_ob, points_coord= list(zip(xs,ys)))
    features = points_object.points
    return features,xs,ys

def read_rectengular_patch_coord(main_dir,palimpsest_name,folio_name,patch_name):
    """
    Reads the page coordinates from a given directory structure and returns the coordinates of the bounding box of the page content.

    Parameters:
    main_dir (str): The main directory path where the data is stored.
    palimpsest_name (str): The name of the palimpsest.
    folio_name (str): The name of the folio.

    Returns:
    tuple: A tuple containing four integers: row_start, row_end, col_start, col_end which represent the coordinates of the bounding box of the page content.
    """
    page_mask_page = os.path.join(main_dir,palimpsest_name,folio_name,"mask",
                                  f"{folio_name}-{patch_name}_black.png")
    page_mask = io.imread(page_mask_page,as_gray=True)
    if np.max(page_mask)>1:
        page_mask = page_mask/np.max(page_mask)
    # Ensure mask is binary
    page_mask[page_mask >= 0.5] = 1
    page_mask[page_mask < 0.5] = 0
    page_mask = 1-page_mask
    row = np.sum(page_mask,axis=1)
    col = np.sum(page_mask,axis=0)
    row_idx_pos = np.argwhere(row>0.5)
    row_start = np.min(row_idx_pos)
    row_end = np.max(row_idx_pos)
    col_idx_pos = np.argwhere(col>0.5)
    col_start = np.min(col_idx_pos)
    col_end = np.max(col_idx_pos)
    return col_start,row_start,col_end+1,row_end+1