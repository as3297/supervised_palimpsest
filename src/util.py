import json
import numpy as np

def save_json(fpath,d):
    # Write the dictionary to a JSON file
    with open(fpath, 'w') as json_file:
        json.dump(d, json_file, indent=4)

def read_json(fpath):
    with open(fpath, 'r') as json_file:
        d = json.load(json_file)
    return d

def read_band_list(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    bands = []
    for line in lines:
        bands.append(line.strip("\n"))
    return bands

def read_max_vals(fpath_max_val,bands_list):
    """

    :param fpath_max_val:
    :param band_list:
    :return:
    """
    d = read_json(fpath_max_val)
    max_vals = np.zeros([len(bands_list)])
    if len(bands_list)==0:
        bands_list = list(d.keys())
        bands_list = order_band_list(bands_list)

    for band_idx, band_name in enumerate(bands_list):
        max_vals[band_idx] = d[band_name]
    return max_vals

def order_band_list(bands):
    ordered_bands = []
    list_of_band_idxs = []
    for band in bands:
        list_of_band_idxs.append(int(band[-4:-2]))

    for band_idx in sorted(list_of_band_idxs):
        for band_name in bands:
            print("Band name:{},idx:{}".format(band_name, band_name[-4:-2]))
            if int(band_name[-4:-2]) == band_idx:
                ordered_bands.append(band_name + "\n")
                break
    return ordered_bands