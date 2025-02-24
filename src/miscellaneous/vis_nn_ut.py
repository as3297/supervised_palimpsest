import sys
import os

from src.temp.test_spectralon_target import folio_names

# Add the root directory (one level up) to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from util import read_json
import os
from skimage import io
import numpy as np



if __name__ == "__main__":

    root_dir = r"/projects/palimpsests" #r"D:"
    palimpsest_name = "Verona_msXL"
    main_data_dir = os.path.join(root_dir, palimpsest_name)
    folio_name = "msXL_335v_b"
    modality = "M"
    class_name = "undertext"
    n = 3
    box = None
    fpath = os.path.join(root_dir,palimpsest_name,folio_name,f"euclid_nn_{n}.json")
    dict_list = read_json(fpath)
    fpath = os.path.join(root_dir,palimpsest_name,folio_name,"mask",folio_name+"-"+class_name+"_black.png")
    im = io.imread(fpath, as_gray=True)
    im_zero = np.zeros(im.shape)

    dict_list=dict_list[folio_name]
    dist_l, xs_l,ys_l, folio_name_l = [],[],[],[]

    for dict in dict_list:
        for folio_name in dict.keys():
            dist = dist[folio_name]["dist"]
            xs_ut = dist[folio_name]["xs_ut"]
            ys_ut = dist[folio_name]["ys_ut"]
            xs = dist[folio_name]["xs"]
            ys = dist[folio_name]["ys"]
            dist_l.append(dist)
            xs_l.append(xs)
            ys_l.append(ys)
            folio_name = np.repeat(np.array([folio_name]*len(dist))[:, np.newaxis], repeats=n, axis=1)
            folio_name_l.append(folio_name)
            folios_names.append(folio_name)

    dist = np.concatenate(dist_l,axis=1)
    xs = np.concatenate(xs_l,axis=1)
    ys = np.concatenate(ys_l,axis=1)
    folio_names = np.concatenate(folio_name_l,axis=1)

    n_nn_idx = np.argsort(dist, axis=1)[:, 0:n]


    dist = np.take_along_axis(dist, n_nn_idx, axis=1)
    xs = np.take_along_axis(xs, n_nn_idx, axis=1)
    ys = np.take_along_axis(ys, n_nn_idx, axis=1)
    folio_names = np.take_along_axis(folio_names, n_nn_idx, axis=1)


    for i in range(len(dist)):
        new_dict[i] = {"dist":dist[i],"xs_ut":xs_ut[i],"ys_ut":ys_ut[i],"xs":xs[i],"ys":ys[i],"folio_names":folio_names[i]}




