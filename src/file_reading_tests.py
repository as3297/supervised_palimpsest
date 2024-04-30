from read_msi_image import PointsfromMSI_PIL, FragmentfromMSI_PIL, ImageCubePILobject
from util import  read_band_list, read_max_vals
from matplotlib import pyplot as plt
import numpy as np
from training import data_from_file

def generate_coord_inside_bbox(x1,y1,width,height):
    points_coords = []
    for j in range(height):
        for i in range(width):
            points_coords.append([i+x1,j+y1])
    return points_coords

def fragment_generation_test():
    x1, y1 = [60, 1197]
    image_dir = r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated"
    band_list_path = r"C:\Data\PhD\palimpsest\Victor_data\band_list.txt"
    fpath_max_val = r"C:\Data\PhD\palimpsest\Victor_data\bands_max_val.json"
    width, height = 1000, 500
    points_coord = generate_coord_inside_bbox(x1,y1,width,height)
    bands = read_band_list(band_list_path)
    max_vals = read_max_vals(fpath_max_val,bands)
    im_pil_ob = ImageCubePILobject(image_dir,bands,0)
    points_ob = PointsfromMSI_PIL(im_pil_ob,max_vals,points_coord)
    band_idx = 15
    frag_im = points_ob.points[band_idx].reshape([height,width])
    frag_obj = FragmentfromMSI_PIL(im_pil_ob,max_vals,[x1,y1,x1+width,y1+height])
    res = np.sum(frag_obj.ims_img[band_idx] - frag_im)
    if res>0:
        raise ValueError("Fragment and reconstracted fragment from points are not the same")
    plt.figure("Fragment")
    plt.imshow(frag_obj.ims_img[band_idx],cmap="gray")

    plt.figure("Fragment from points")
    plt.imshow(frag_im,cmap="gray")
    plt.show()

if __name__=="__main__":
    im_path_ut = r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\maks\msXL_315r_b-undertext_black.png"
    im_path_notut = r"C:\Data\PhD\palimpsest\Victor_data\msXL_315r_rotated\maks\msXL_315r_b-not_undertext.png"
    data_from_file(im_path_ut,im_path_notut)