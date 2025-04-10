import numpy as np
from src.read_data import read_subset_features,read_subset_features_patches
import os


main_data_dir = r"D:"
palimpsest_name = r"Verona_msXL"
main_dir = os.path.join(main_data_dir, palimpsest_name)
classes_dict = {"undertext": 1, "not_undertext": 0}
modalities = ["M"]
folios_train = ["msXL_335v_b"]
win = 12
box = None
features_ut,xs_ut,ys_ut = read_subset_features_patches(main_dir,folio_name,class_name,modalities,win,box)