import sys
import os

# Add the root directory (one level up) to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from util import read_pickle
import os
from skimage import io
import numpy as np


def assign_pixel_indices(xs, ys):
    """
    Function to assign an index to every pixel based on its coordinates,
    starting from the lowest values in a sorted manner.

    :param xs: 2D array-like meshgrid of x coordinates
    :param ys: 2D array-like meshgrid of y coordinates
    :return: 2D array of indices assigned to each pixel
    """

    # Combine the coordinates into a structured array for sorting
    indexes = np.lexsort((ys,xs))
    xs = xs[indexes]
    ys = ys[indexes]

    return xs,ys, indexes


# Generate an image where each pixel in the dictionary gets a gradient color based on its pixel index
def create_gradient_colored_image(dict_folio, image_shape):
    """
    Create an image where each pixel in the dictionary gets a gradient color based on its pixel index.

    :param new_dict: Dictionary containing pixel data and indices.
    :param image_shape: Tuple representing the shape of the image (height, width).
    :return: A gradient colored image (H, W, 3).
    """
    # Initialize a black RGB image
    gradient_image = np.zeros((*image_shape, 3), dtype=np.uint8)

    # Total number of unique pixels
    total_pixels = image_shape[0] * image_shape[1]

    # Iterate over the dictionary to assign gradient colors
    xs = dict_folio["xs"]
    ys = dict_folio["ys"]
    idx = dict_folio["idx"]

    # Calculate gradient RGB color based on pixel indices normalized to the range (0, 255)
    idx = idx/ len(idx)

    r = (255 * np.maximum(0, np.minimum(1, 4 * np.abs(idx - 0.75) - 1))).astype(np.uint8)
    g = (255 * np.maximum(0, np.minimum(1, 4 * np.abs(idx - 0.5) - 1))).astype(np.uint8)
    b = (255 * np.maximum(0, np.minimum(1, 4 * np.abs(idx - 0.25) - 1))).astype(np.uint8)

    color = np.stack([r, g, b],axis=1)

    # Assign the gradient RGB color to the specified pixels
    for i in range(len(xs)):
        #print(gradient_image[int(ys[i]), int(xs[i]),:].shape)
        #print(color[i,:].shape)
        gradient_image[int(ys[i]), int(xs[i]),:] = color[i,:]

    return gradient_image

if __name__ == "__main__":

    root_dir = r"D:"
    palimpsest_name = "Verona_msXL"
    main_data_dir = os.path.join(root_dir, palimpsest_name)
    ut_folio_name = "msXL_335v_b"
    modality = "M"
    class_name = "undertext"
    n = 3
    box = None
    distances_dir = os.path.join(root_dir,palimpsest_name,ut_folio_name,"distances")

    for fpath in ["msXL_335v_b_msXL_318r_b_euclid_nn_3.pkl",
                  "msXL_335v_b_msXL_315v_b_euclid_nn_3.pkl",
                  "msXL_335v_b_msXL_335v_b_euclid_nn_3.pkl"]:
                  #os.listdir(distances_dir):
        if not fpath.endswith(".pkl"):
            continue
        #fpath = os.path.join(distances_dir,ut_folio_name+"_"+folio_name+"_"+f"euclid_nn_{n}.pkl")
        folio_name = "_".join(fpath.split("_")[3:6])
        print(
            f"Processing file: {fpath} for folio: {folio_name}")
        dict_ut = read_pickle(os.path.join(distances_dir,fpath))

        im = io.imread(os.path.join(main_data_dir,folio_name,"mask", folio_name+"-undertext_black.png"), as_gray=True)
        im_zero = np.zeros(im.shape)

        dist_l, xs_l,ys_l, folio_name_l = [],[],[],[]
        xs_ut = dict_ut["xs_ut"]
        ys_ut = dict_ut["ys_ut"]

        dist = dict_ut["dist"]
        xs = dict_ut["xs"]
        ys = dict_ut["ys"]
        print("Non zero xs elements: ", np.count_nonzero(xs))
        folio_name = np.repeat(np.array([folio_name] * len(dist))[:, np.newaxis], repeats=n, axis=1)

    dist_l.append(dist)
    xs_l.append(xs)
    ys_l.append(ys)
    folio_name_l.append(folio_name)

    dist = np.concatenate(dist_l,axis=1)
    xs = np.concatenate(xs_l,axis=1)
    ys = np.concatenate(ys_l,axis=1)
    folio_names = np.concatenate(folio_name_l,axis=1)

    n_nn_idx = np.argsort(dist, axis=1)[:, 0:n]


    dist = np.take_along_axis(dist, n_nn_idx, axis=1)
    xs = np.take_along_axis(xs, n_nn_idx, axis=1)
    ys = np.take_along_axis(ys, n_nn_idx, axis=1)
    folio_names = np.take_along_axis(folio_names, n_nn_idx, axis=1)

    xs_ut,ys_ut,indexes = assign_pixel_indices(np.array(xs_ut),np.array(ys_ut))
    indexes = indexes[:len(xs)]
    # Example usage
    image_shape = im.shape  # Use the shape of `im` as a reference
    new_dict = {"xs":xs_ut,"ys":ys_ut,"idx":indexes}
    colored_image = create_gradient_colored_image(new_dict, image_shape)
    # Save the generated image
    output_path = os.path.join(distances_dir, "org_coordinates.png")
    io.imsave(output_path, colored_image)
    print(f"Colored image saved at: {output_path}")
    dict_ut = {"xs_ut": xs_ut, "ys_ut": ys_ut, "idx": indexes,"dist":dist,"xs":xs[indexes],"ys":ys[indexes],"folio_names":folio_names[indexes]}

    for i in range(len(dist[0])):
        folio_names = set(dict_ut["folio_names"][:,i])
        for folio_name in folio_names:
            idxs = np.argwhere(dict_ut["folio_names"][:,i] == folio_name)
            new_dict = {"xs":np.concatenate(dict_ut["xs"][idxs,i]),"ys":np.concatenate(dict_ut["ys"][idxs,i]),"idx":np.concatenate(dict_ut["idx"][idxs])}
            colored_image = create_gradient_colored_image(new_dict, image_shape)
            output_path = os.path.join(distances_dir, folio_name+f"_coordinates_nn{i}.png")
            io.imsave(output_path, colored_image)



