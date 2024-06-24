from read_msi_data_as_array import PointsfromMSI_PIL, FragmentfromMSI_PIL
import numpy as np
import os
from pil_image_cube import ImageCubePILobject
from util import read_max_vals,read_band_list,generate_coord_inside_bbox
from dataset import sublist_of_bands,read_features_labels,shuffle_dataset_split,read_bboxs
from util import read_json
from matplotlib import pyplot as plt
from PIL import Image
from read_pixel_coord import points_coord_in_bbox,ClassCoord
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
def fragment_generation_test():
    folio_name = r"msXL_315r_b"
    osp = os.path.join
    image_dir = osp(r"C:\Data\PhD\palimpsest\Victor_data",folio_name)
    band_list_path = r"C:\Data\PhD\palimpsest\Victor_data\band_list.txt"
    fpath_max_val = r"C:\Data\PhD\palimpsest\Victor_data\bands_max_val.json"

    bbox_fpath = osp(r"C:\Data\PhD\palimpsest\Victor_data",folio_name,"dataset_split.json")
    bbox_dict = read_json(bbox_fpath)
    bbox = read_bboxs("val", bbox_dict)
    height = bbox[3]-bbox[1]
    width = bbox[2]-bbox[0]
    x1, y1 = bbox[0],bbox[1]
    points_coord = generate_coord_inside_bbox(x1,y1,width,height)
    bands = read_band_list(band_list_path)
    band_idx = 14
    print("Band name {}".format(bands[band_idx]))
    max_vals = read_max_vals(fpath_max_val,bands)
    im_pil_ob = ImageCubePILobject(image_dir,folio_name,bands,0)
    points_ob = PointsfromMSI_PIL(im_pil_ob,max_vals,points_coord)

    frag_im = points_ob.points[:,band_idx].reshape([height,width])
    frag_obj = FragmentfromMSI_PIL(im_pil_ob,max_vals,[x1,y1,x1+width,y1+height])
    res = np.sum(frag_obj.ims_img[band_idx] - frag_im)
    if res>0:
       raise ValueError("Fragment and reconstracted fragment from points are not the same")
    plt.figure("Fragment")
    plt.imshow(frag_obj.ims_img[band_idx],cmap="gray")

    plt.figure("Fragment from points")
    plt.imshow(frag_im,cmap="gray")
    plt.show()


def test_load_data_points():
  """

  :return:
  [],[],[] - first list train data [features,labels] for undertext,
            second list train data [features,labels] for nonundertext,
            third list validation data [features,labels] for shuffled undertext  and nonundertext

  """
  osp = os.path.join
  main_dir = r"C:\Data\PhD\palimpsest\Victor_data"
  band_list_path = osp(main_dir,"band_list.txt")
  bands = read_band_list(band_list_path)
  bands = sublist_of_bands(bands,"M")
  max_val_path = osp(main_dir,"bands_max_val.json")
  max_vals = read_max_vals(max_val_path,bands)
  image_dir_path = osp(main_dir,r"msXL_315r_rotated")
  bbox_fpath = osp(image_dir_path,"dataset_split.json")
  folioname = r"msXL_315r_b"
  ut_mask_path = osp(image_dir_path,"mask","msXL_315r_b-undertext_black.png")
  nonut_mask_path = osp(image_dir_path,"mask",r"msXL_315r_b-not_undertext.png")
  im_msi_pil_ob = ImageCubePILobject(image_dir_path,folioname,bands,0)
  bbox_dict = read_json(bbox_fpath)
  trainset_ut,trainset_nonut = read_features_labels(bbox_dict,im_msi_pil_ob,
                                  ut_mask_path, nonut_mask_path, max_vals, "train")
  sidx = 0
  print("Step 1 extract: ut sample {}\n,features {}\n, label {}".format(sidx,trainset_ut[0][sidx],trainset_ut[1][sidx]))
  print("Step 1 extract : nonut sample {}\n,features {}\n, label {}".format(sidx,trainset_nonut[0][sidx],trainset_nonut[1][sidx]))

  trainset_ut,ut_idxs = shuffle_dataset_split(trainset_ut,True)
  ut_idx = np.argwhere(ut_idxs==100)
  trainset_nonut,nonut_idxs = shuffle_dataset_split(trainset_nonut,True)
  nonut_idx = np.argwhere(nonut_idxs==100)
  print("Step 2 shuffle separately: ut sample {}\n,features {}\n, label {}".format(sidx,trainset_ut[0][ut_idx],trainset_ut[1][ut_idx]))
  print("Step 2 shuffle separately: nonut sample {}\n,features {}\n, label {}".format(sidx,trainset_nonut[0][nonut_idx],trainset_nonut[1][nonut_idx]))
  print("Step 3 before equalization: ut sample {}\n,features {}\n, label {}".format(sidx,trainset_ut[0][sidx],trainset_ut[1][sidx]))
  print("Step 3 before equalization: nonut sample {}\n,features {}\n, label {}".format(sidx,trainset_nonut[0][sidx],trainset_nonut[1][sidx]))
  nb_ut_samples = len(trainset_ut[1])-1
  trainset = equilize_nb_dataset_points(trainset_ut,trainset_nonut)
  print("Step 3 equalize: ut sample {}\n,features {}\n, label {}".format(sidx,trainset[0][sidx],trainset[1][sidx]))
  print("Step 3 equalize: nonut sample {}\n,features {}\n, label {}".format(sidx,trainset[0][nb_ut_samples+sidx+1],trainset[1][nb_ut_samples+sidx+1]))
  

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

def test_points_coord_in_box():
    fpath = r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\mask\msXL_315r_b-not_undertext_more.png"
    mask = LabeledIm(fpath, 0).im
    bbox = [60, 1197, 4680, 2288]
    xs, ys, nb_coords = points_coord_in_bbox(fpath, bbox)
    restored_mask = np.ones_like(mask)
    restored_mask[ys, xs] = 0
    plt.figure("Restored mask")
    plt.imshow(restored_mask, cmap="gray")
    plt.show()

def test_points_coord_in_bbox():
    osp = os.path.join
    folio_name = r"msXL_315r_b"
    main_dir = r"C:\Data\PhD\palimpsest\Victor_data"
    image_dir_path = osp(main_dir, folio_name)
    bbox_fpath = osp(image_dir_path, "dataset_split.json")
    bbox_dict = read_json(bbox_fpath)
    split = "val"
    bbox = read_bboxs(split,bbox_dict)
    ut_mask_path = osp(image_dir_path, "mask",folio_name + r"-undertext_black.png")
    nonut_mask_path = osp(image_dir_path, "mask",folio_name+ r"-not_undertext.png")
    rgb_palimp = osp(image_dir_path,"mask",r"rgb-compose.png")
    coords_nonut_x,coords_nonut_y,_ = points_coord_in_bbox(nonut_mask_path,bbox)
    coords_ut_x,coords_ut_y,_ = points_coord_in_bbox(ut_mask_path,bbox)
    mask = LabeledIm(rgb_palimp, 0).im
    mask[coords_nonut_y, coords_nonut_x] = 1
    mask[coords_ut_y, coords_ut_x] = 1
    save_dir = osp(r"C:\Data\PhD\palimpsest\Victor_data", folio_name,"miscellaneous")
    io.imsave(osp(save_dir,"split_"+split+".png"),(mask * 255).astype(np.uint8))


if __name__=="__main__":
    #im_path_ut = r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\maks\msXL_315r_b-undertext_black.png"
    #im_path_notut = r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\maks\msXL_315r_b-not_undertext.png"
    #fragment_generation_test()
    test_points_coord_in_bbox()