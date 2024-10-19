from msi_data_as_array import PointsfromMSI_PIL, FragmentfromMSI_PIL
import numpy as np
import os
from pil_image_cube import ImageCubePILobject
from util import read_band_list,generate_coord_inside_bbox,read_split_box_coord
from util import read_json
from matplotlib import pyplot as plt
from PIL import Image
from pixel_coord import points_coord_in_bbox,ClassCoord
from skimage import io



class LabeledIm:
    def __init__(self,fpath,rotate_angle):
        """
        :param image_dir: directory with tif image of palimpsest
        :param band_list: list of bands
        :param coord: (left, upper, right, lower) tuple of bounding box coordinates
        """
        self.fpath = fpath
        self.rotate_angle = rotate_angle
        self.im = self.read_file()

    def read_file(self):
        """
        Read image mask
        :return:
        coords: [[row_0,col_0],...,[row_i,col_i]]
        """
        with Image.open(self.fpath) as im:
            im_mode = im.mode
            if self.rotate_angle > 0:
                rotation = eval("Image.ROTATE_{}".format(self.rotate_angle))
                im = im.transpose(rotation)
            im = np.array(im)
        if im_mode == "RGBA":
            im = im[:,:,3]
            im = np.amax(im) - im
        im = im / np.amax(im)
        return im

def read_points(main_dir,folio_name,points_coord,bands):

    return points_ob



def fragment_generation_test(main_dir,folio_name):
    osp = os.path.join
    band_list_path = osp(main_dir, r"band_list.txt")
    bands = read_band_list(band_list_path,"M")
    bbox_fpath = osp(main_dir, folio_name, "dataset_split.json")
    bbox_dict = read_json(bbox_fpath)
    bbox = read_split_box_coord("val", bbox_dict)
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    x1, y1 = bbox[0], bbox[1]
    points_coord = generate_coord_inside_bbox(x1, y1, width, height)
    im_pil_ob = ImageCubePILobject(main_dir,folio_name,bands,0)
    points_ob = PointsfromMSI_PIL(im_pil_ob,points_coord)
    band_idx = 14
    print("Band name {}".format(bands[band_idx]))
    frag_im = points_ob.points[:,band_idx].reshape([height,width])
    frag_obj = FragmentfromMSI_PIL(im_pil_ob,[x1,y1,x1+width,y1+height])
    res = np.sum(frag_obj.ims_img[band_idx] - frag_im)
    if res>0:
       raise ValueError("Fragment and reconstracted fragment from points are not the same")
    plt.figure("Fragment")
    plt.imshow(frag_obj.ims_img[band_idx],cmap="gray")

    plt.figure("Fragment from points")
    plt.imshow(frag_im,cmap="gray")
    plt.show()


def test_read_Class_coord():
    fpath = r"C:\Data\PhD\palimpsest\Victor_data\msXL_319r_b\mask\msXL_319r_b-spectralon.png"
    coords_reverse = ClassCoord(fpath,0).coords
    coords = list(zip(list(zip(*coords_reverse))[1],list(zip(*coords_reverse))[0]))
    with Image.open(fpath) as im_pil:
        pixel_vals = list(map(im_pil.getpixel, coords_reverse))
    mask = LabeledIm(fpath,0).im
    restored_mask = np.ones_like(mask)
    restored_pixel_val = np.ones_like(mask)

    rows, cols = map(list, zip(*coords))
    restored_mask[rows,cols]=0
    restored_pixel_val[rows,cols] = pixel_vals


    plt.figure("Original mask")
    plt.imshow(mask, cmap = "gray")

    plt.figure("Restored mask")
    plt.imshow(restored_mask, cmap = "gray")

    plt.figure("Restored pixel vals")
    plt.imshow(restored_pixel_val, cmap="gray")
    plt.show()



def test_points_coord_in_bbox(main_dir,folio_name):
    osp = os.path.join
    image_dir_path = osp(main_dir, folio_name)
    bbox_fpath = osp(image_dir_path, "dataset_split.json")
    bbox_dict = read_json(bbox_fpath)
    split = "val"
    bbox = read_bboxs(split,bbox_dict)
    ut_mask_path = osp(image_dir_path, "mask",folio_name + r"-undertext_black.png")
    nonut_mask_path = osp(image_dir_path, "mask",folio_name+ r"-not_undertext_black.png")
    rgb_palimp = osp(image_dir_path,"mask",r"rgb-compose.png")
    coords_nonut_x,coords_nonut_y,_ = points_coord_in_bbox(nonut_mask_path,bbox)
    coords_ut_x,coords_ut_y,_ = points_coord_in_bbox(ut_mask_path,bbox)
    mask = LabeledIm(rgb_palimp, 0).im
    mask[coords_nonut_y, coords_nonut_x] = 1
    mask[coords_ut_y, coords_ut_x] = 1
    save_dir = osp(main_dir, folio_name,"miscellaneous")
    io.imsave(osp(save_dir,"split_"+split+".png"),(mask * 255).astype(np.uint8))


if __name__=="__main__":
    main_dir = r"c:\Data\PhD\palimpsest\Victor_data\Paris_Coislin"
    folio_name = r"Par_coislin_393_054r"

    fragment_generation_test(main_dir,folio_name)
    #test_points_coord_in_bbox(main_dir,folio_name)