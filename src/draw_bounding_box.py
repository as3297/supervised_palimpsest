import os
import cv2
import numpy as np
from read_files import ReadImageCube
from bounding_box import BoundingBoxWidget
from util import save_json,read_json

def show_im_thumbnail(image_dir):
    """Show image thumbnail band by band"""
    img = ReadImageCube(image_dir,[],270,[0,0,500,500])
    scale_ratio = 10
    for idx,band in enumerate(img.band_list):
            im = img.thumbnail(idx,scale_ratio)
            im = im / np.amax(im)
            cv2.imshow(band,im)
            cv2.waitKey(0)

def select_im_fragment(image_dir):
    """Get coordinates of selected rectangular image fragments"""
    img = ReadImageCube(image_dir, [], 270)
    scale_ratio = 10
    im = img.thumbnail(13, scale_ratio)
    #im = im / np.amax(im)
    boundingbox_widget = BoundingBoxWidget(im)
    cv2.imshow('image', boundingbox_widget.show_image())
    cv2.waitKey(0)
    boxes = boundingbox_widget.boxes_coordinates
    for box_idx,box in enumerate(boxes):
        box["box_"+str(box_idx)] = tuple(map(mult_by_scaler(scale_ratio),box["box_"+str(box_idx)]))
    return boxes

def mult_by_scaler(scaler):
    def mult(x):
        return x*scaler
    return mult

def check_boxes():
    image_dir = r"C:\Data\PhD\palimpsest\Victor_data"
    json_file = os.path.join(image_dir, "coord_forsvm.json")
    d = read_json(json_file)
    bg_coords = d['background']
    img = ReadImageCube(image_dir, [], 270)
    scale_ratio = 10
    band_idx = 14
    im = img.thumbnail(band_idx, scale_ratio)
    path = os.path.join(image_dir,img.band_list[band_idx]+".tif")
    for box_idx, box_coord in enumerate(bg_coords):
        box = box_coord["box_"+str(box_idx)]

        row_t = int(box[1]/10)
        row_b = int(box[3]/10)
        col_t = int(box[0]/10)
        col_b = int(box[2]/10)
        im[row_t:row_b,col_t] = 0
        im[row_t:row_b,col_b] = 0
        im[row_t,col_t:col_b] = 0
        im[row_b,col_t:col_b] = 0
        im_box = img.read_image(path, True, 1, box, img.max_values[band_idx])
        cv2.imshow("box_"+str(box_idx),im_box)
    cv2.imshow("boxes",im)
    cv2.waitKey(0)

if __name__ == '__main__':
    image_dir = r"C:\Data\PhD\palimpsest\Victor_data"
    coord = select_im_fragment(image_dir)
    fpath = os.path.join(image_dir,"coord_forsvm.json")
    save_json(fpath,{"background":coord})
    check_boxes()