import copy

from src.msi_data_as_array import FullImageFromPILImageCube
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import os
from src.read_data import read_subset_features
from src.dataset import read_features,stack_features_labels
from src.pil_image_cube import ImageCubePILobject

root_dir = r"D:"
palimpsest_name = "Verona_msXL"
main_dir = os.path.join(root_dir,palimpsest_name)
folio_name = r"msXL_335v_b"
modalities = "M"
box = None
class_name = "undertext"

def plot_class_image_vs_real_image():
    features_ut,xs_ut,ys_ut = read_subset_features(main_dir,folio_name,class_name,modalities,box)
    features_dict = read_features(main_dir, folio_name, class_name, modalities,box)
    dataset = stack_features_labels(features_dict)


    msi_pil_obj = ImageCubePILobject(main_dir,folio_name,modalities,0)
    msi_im = FullImageFromPILImageCube(msi_pil_obj).ims_img
    msi_im_subst = copy.deepcopy(msi_im)
    msi_im_subst[ys_ut,xs_ut] = features_ut

    im_ut = io.imread(os.path.join(root_dir,palimpsest_name,folio_name,"mask",folio_name+"-undertext_black.png"))[:,:,0]
    im_shape = im_ut.shape
    im = np.ones(im_shape)
    im[ys_ut,xs_ut] = 0
    plt.figure()
    plt.suptitle("Check undertext coordinates")
    plt.subplot(1,2,1)
    plt.imshow(im,cmap="gray")
    plt.title("Undertext restored from coordinates")
    plt.subplot(1,2,2)
    plt.imshow(im_ut,cmap="gray")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap="gray")
    plt.title("Undertext")
    plt.subplot(1, 2, 2)
    plt.imshow(im_ut, cmap="gray")


if __name__ == '__main__':
    unittest.main()
