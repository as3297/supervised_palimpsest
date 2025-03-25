import copy

from src.msi_data_as_array import FullImageFromPILImageCube,PatchesfromMSI_PIL,FragmentfromMSI_PIL
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import os
from src.read_data import read_subset_features
from src.dataset import read_features,stack_features_labels
from src.pil_image_cube import ImageCubePILobject
from PIL import Image

root_dir = r"D:"
palimpsest_name = "Verona_msXL"
main_dir = os.path.join(root_dir,palimpsest_name)
folio_names = [r"msXL_335v_b"]
modalities = ["M"]
box = None
class_name = "undertext"
class_dict = {"undertext":1}
win = 0

def read_mask(root_dir,palimpsest_name,folio_name,class_name):
    with Image.open(os.path.join(root_dir,palimpsest_name,folio_name,"mask",folio_name+f"-{class_name}_black.png")).convert('L') as im:
        im_ut = np.array(im)
    im = im/np.max(im)
    im = im>0.5
    im = im.astype(int)
    return im
def calculate_zero_elements(array):
    """
    Calculate the number of zero elements in a given array.

    :param array: NumPy array.
    :return: Integer count of zero elements in the array.
    """
    return np.size(array) - np.count_nonzero(array)

def test_class_image_vs_real_image():
    win=0
    cur_folio_names = [folio_names[0]]
    features_ut,xs_ut,ys_ut = read_subset_features(main_dir,cur_folio_names[0],class_name,modalities,box)
    features_dict = read_features(main_dir, cur_folio_names, class_dict, modalities,box,win)

    msi_pil_obj = ImageCubePILobject(main_dir,cur_folio_names[0],modalities,0)
    msi_im = FullImageFromPILImageCube(msi_pil_obj).ims_img
    msi_im_subst = copy.deepcopy(msi_im)
    msi_im_subst[ys_ut,xs_ut] = features_dict[class_dict[class_name]]

    im_ut = read_mask(root_dir,palimpsest_name,cur_folio_names[0],class_name)
    print("Amount of zero elements in the im_ut image: ",calculate_zero_elements(im_ut))
    im_shape = im_ut.shape
    im = np.ones(im_shape)
    im[ys_ut,xs_ut] = 0


    print("Amount of zero elements in the extracted ut pixels image: ", calculate_zero_elements(im))
    np.testing.assert_array_equal(
        im_ut,
        im,
        err_msg="Unertext coordinates do not match"
    )
    np.testing.assert_array_equal(
        msi_im,
        msi_im_subst,
        err_msg="Features and image do not match"
    )
    print("test_class_image_vs_real_image: OK")

def plot_class_image_vs_real_image():
    win=0
    cur_folio_names = [folio_names[0]]
    features_ut,xs_ut,ys_ut = read_subset_features(main_dir,cur_folio_names[0],class_name,modalities,box)
    features_dict = read_features(main_dir, cur_folio_names, class_dict, modalities,box,win)

    msi_pil_obj = ImageCubePILobject(main_dir,cur_folio_names[0],modalities,0)
    msi_im = FullImageFromPILImageCube(msi_pil_obj).ims_img
    msi_im_subst = copy.deepcopy(msi_im)
    msi_im_subst[ys_ut,xs_ut] = features_dict[class_dict[class_name]]

    im_ut = read_mask(root_dir,palimpsest_name,cur_folio_names[0],class_name)
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
    plt.suptitle("Check undertext coordinates")
    plt.subplot(1, 2, 1)
    plt.imshow(msi_im[:,:,:3])
    plt.title("Original MSI image")
    plt.subplot(1, 2, 2)
    plt.imshow(msi_im_subst[:,:,:3])
    plt.show()

def test_stack_features_labels():
    ut_pixels = 0
    non_ut_pixels = 0
    win=0
    class_dict = {"undertext": 1, "not_undertext": 0}
    folio_names = [r"msXL_335v_b", r"msXL_319v_b"]
    for folio_name in folio_names:
        im_ut = read_mask(root_dir,palimpsest_name,folio_name,"undertext")
        im_non_ut = read_mask(root_dir,palimpsest_name,folio_name,"not_undertext")
        # Count all non-zero elements in im_ut
        ut_pixels += calculate_zero_elements(im_ut)
        non_ut_pixels += calculate_zero_elements(im_non_ut)
    dict_test = {"undertext": ut_pixels, "not_undertext": non_ut_pixels}
    print(f"Number of ut elements: {ut_pixels}")
    print(f"Number of non-ut elements: {non_ut_pixels}")
    features_dict = read_features(main_dir, folio_names, class_dict, modalities,box,win)
    dataset,labels = stack_features_labels(features_dict)

    # Count the occurrences of each unique value in the labels list
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print("Label counts:", label_counts)
    for class_name,class_idx in class_dict.items():
        if label_counts[class_idx] != dict_test[class_name]:
            raise Exception(f"Number of elements in class {class_name} do not match. In stack_features_labels function nb {label_counts[class_idx]},"
                          f"in read_features nb {dict_test[class_name]}")
    print("test_stack_features_labels: OK")

def test_nb_dimenssion():
    class_dict = {"undertext": 1, "not_undertext": 0}
    folio_names = [r"msXL_335v_b",r"msXL_319v_b"]
    win = 0
    features_dict = read_features(main_dir, folio_names, class_dict, modalities,box,win)
    dataset,labels = stack_features_labels(features_dict)
    if len(dataset.shape)<2 or len(dataset.shape)>3:
        raise Exception("Dataset has wrong dimensionality. It should be 2D or 3D")
    if len(labels) != dataset.shape[0]:
        raise Exception("Labels and dataset do not have the same length")
    print("test_nb_dimenssion: OK")

def does_it_align():
    init_point = (1200, 2200)
    win = 200
    nb_points = 2
    hw = win // 2
    points_coord = [(init_point[0] + i*(hw*2+1), init_point[1]) for i in range(nb_points)]
    frag_coord = [init_point[0]-hw,init_point[1]-hw,init_point[0]+nb_points*(2*hw+1)-hw,init_point[1]+(2*hw+1)-hw]
    pil_msi_obj = ImageCubePILobject(main_dir,folio_names[0],modalities,0)
    patches_msi = PatchesfromMSI_PIL(pil_msi_obj, points_coord, win).ims_imgs
    features_dict = {"undertext":patches_msi}
    patches_msi, labels = stack_features_labels(features_dict)
    max_val = np.max(patches_msi)

    # Concatenating patches into one big image
    patches_msi_concat = np.zeros(((hw*2+1),(hw*2+1)*nb_points,pil_msi_obj.nb_bands))

    for (x, y), patch in zip(points_coord, patches_msi):
        x = x - init_point[0]
        y = y - init_point[1]
        patches_msi_concat[y:y+2*hw+1, x:x+2*hw+1,:] = patch
    frag_msi = FragmentfromMSI_PIL(pil_msi_obj , frag_coord).ims_img

    np.testing.assert_array_equal(
        patches_msi_concat,
        frag_msi,
        err_msg="Patches and fragments do not align"
    )
    print("Test concluded")
    patches_msi_concat = patches_msi_concat/max_val
    plt.figure()
    plt.imshow(patches_msi_concat[:,:,-4:-1])
    plt.show()

if __name__ == '__main__':
    #test_class_image_vs_real_image()
    #test_stack_features_labels()
    #test_nb_dimenssion()
    does_it_align()

