import json

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
        bands.append(line.strip("\n")[:-4])
    return bands