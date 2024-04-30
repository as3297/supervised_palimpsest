from PIL import Image, ImageOps
import os
import numpy as np
from matplotlib import pyplot as plt
from read_msi_image import PointsfromMSI_PIL

class ClassCoord:
    def __init__(self,fpath,rotate_angle):
        """
        :param image_dir: directory with tif image of palimpsest
        :param band_list: list of bands
        :param coord: (left, upper, right, lower) tuple of bounding box coordinates
        """
        self.fpath = fpath
        self.rotate_angle = rotate_angle
        self.coords = self.read_file()

    def read_file(self):
        """
        Read coordinates of black pixels from image mask
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
                im = im[:, :, 3]
                im = np.amax(im) - im
        im = im/np.amax(im)
        coords = np.argwhere(im<0.1)
        return coords

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
        Read coordinates of black pixels from image mask
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

def points_coord_in_bbox(fpath,bbox):
    """
    Read points from the mask in the range of bounding box
    :param fpath - path to image mask
    :param bbox - [top,left,bottom,right] bbox coordinates that defines the range of dataset
    """
    coords = ClassCoord(fpath, 0).coords
    ys, xs = map(list, zip(*coords))

    nb_coords = len(coords)
    idx_start = 0
    idx_end = 0
    for idx_x in range(nb_coords):
        if xs[idx_x]>bbox[0]:
          idx_start = idx_x
          break
    for idx_y in range(idx_start,nb_coords):
        if ys[idx_y]>bbox[1]:
          idx_start = idx_y
          break
    for idx_x in range(nb_coords-1,idx_start-1,-1):
        if xs[idx_x]<bbox[2]:
          idx_end = idx_x
          break
    for idx_y in range(nb_coords-1,idx_start-1,-1):
        if ys[idx_y]<bbox[3]:
          idx_end = idx_y
          break
    nb_coords = idx_end-idx_start
    return xs[idx_start:idx_end], ys[idx_start:idx_end], nb_coords


def read_points_class(fpath_image_mask,im_msi_pil_ob,bbox,label,max_vals):
    xs, ys, nb_coords = points_coord_in_bbox(fpath_image_mask, bbox)
    labels = [label]*nb_coords
    points_object = PointsfromMSI_PIL(im_msi_pil_ob, max_vals, zip(xs,ys))
    features = points_object.points
    return features,labels

def test_read_Class_coord():
    fpath = r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\maks\msXL_315r_b-undertext_black.png"
    coords = ClassCoord(fpath,0).coords
    mask = LabeledIm(fpath,0).im
    restored_mask = np.ones_like(mask)
    rows, cols = map(list, zip(*coords))
    restored_mask[rows,cols]=0

    plt.figure("Original mask")
    plt.imshow(mask, cmap = "gray")

    plt.figure("Restored mask")
    plt.imshow(restored_mask, cmap = "gray")
    plt.show()

def test_points_coord_in_box():
    fpath = r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\maks\msXL_315r_b-undertext_black.png"
    mask = LabeledIm(fpath, 0).im
    bbox = [60, 1197, 4680, 2288]
    xs, ys, nb_coords = points_coord_in_bbox(fpath, bbox)
    restored_mask = np.ones_like(mask)
    restored_mask[ys, xs] = 0
    plt.figure("Restored mask")
    plt.imshow(restored_mask, cmap="gray")
    plt.show()


if __name__=="__main__":
    test_points_coord_in_box()