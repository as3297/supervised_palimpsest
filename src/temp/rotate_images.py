import os

import numpy as np
from skimage import io,transform
from matplotlib import pyplot as plt


maindir = r"d:\Verona_msXL"
folioname = r"msXL"

banddict={}
for fname in os.listdir(os.path.join(maindir,folioname)):
    if "tif" in fname:
        fpath = os.path.join(maindir,folioname,fname)
        im0 = io.imread(fpath,as_gray=True)
        im1 = transform.rotate(im0,180,resize=True,preserve_range=True).astype(np.uint16)
        io.imsave(fpath,im1)
