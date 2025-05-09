import sys
import os

# Add the root directory (one level up) to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from src.util import read_pickle,save_json
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

    r = (255 * (idx * 0.8 + 0.1)).astype(np.uint8)
    g = (255 * (idx * 0.7 + 0.2)).astype(np.uint8)
    b = (255 * (idx * 0.6 + 0.3)).astype(np.uint8)

    color = np.stack([r, g, b],axis=1)

    # Assign the gradient RGB color to the specified pixels
    for i in range(len(xs)):
        #print(gradient_image[int(ys[i]), int(xs[i]),:].shape)
        #print(color[i,:].shape)
        gradient_image[int(ys[i]), int(xs[i]),:] = color[i,:]
    return gradient_image

def crop_coord(dict_ut,box=None):
    if box is None:
        return dict_ut
    xs_ut = np.array(dict_ut["xs_ut"])
    ys_ut = np.array(dict_ut["ys_ut"])
    xs_ut_coord = np.argwhere(np.logical_and(xs_ut>box[0],xs_ut<(box[2]+box[0])))
    ys_ut_coord = np.argwhere(np.logical_and(ys_ut>box[1], ys_ut<(box[3]+box[1])))

    # Find the overlap between xs_ut_coord and ys_ut_coord
    overlapping_coords = np.intersect1d(np.squeeze(xs_ut_coord), np.squeeze(ys_ut_coord))

    xs_ut = xs_ut[overlapping_coords]
    ys_ut = ys_ut[overlapping_coords]
    dist = np.array(dict_ut["dist"])[overlapping_coords]
    xs = np.array(dict_ut["xs"])[overlapping_coords]
    ys = np.array(dict_ut["ys"])[overlapping_coords]

    return {"xs_ut":xs_ut,"ys_ut":ys_ut,"dist":dist,"xs":xs,"ys":ys}


# Filter all the xs and ys coordinates that are further than a radius from the xs_ut and ys_ut
def filter_coordinates_within_radius(xs, ys, xs_ut, ys_ut, radius):
    """
    Filters the coordinates (xs, ys) and returns only those which are within a given radius
    from the central points (xs_ut, ys_ut).

    :param xs: Array of x-coordinates to filter
    :param ys: Array of y-coordinates to filter
    :param xs_ut: Array of x-coordinates of central points
    :param ys_ut: Array of y-coordinates of central points
    :param radius: Radius to determine the maximum allowed distance
    :return: Filtered xs and ys arrays
    """
    # Calculate distances between each (xs, ys) point and the central points (xs_ut, ys_ut)
    distances = np.sqrt((xs - xs_ut[:, np.newaxis]) ** 2 + (ys - ys_ut[:, np.newaxis]) ** 2)

    # Find indices where distances are less than or equal to the radius
    valid_indices = np.any(distances <= radius, axis=0)

    # Filter the xs and ys arrays based on valid indices


    return valid_indices
def filter_dict_withing_radius(dict,xs_ut,ys_ut,radius,image_shape):
    valid_indices = filter_coordinates_within_radius(new_dict["xs"], new_dict["ys"], xs_ut,ys_ut, radius)
    xs_ut = xs_ut[valid_indices]
    ys_ut = ys_ut[valid_indices]
    colored_image = create_gradient_colored_image({"xs":xs_ut,"ys":ys_ut,"idx":valid_indices}, image_shape)
    output_path = os.path.join(distances_dir, folio_name + f"_filtered_coordinates_org.png")
    io.imsave(output_path, colored_image)




if __name__ == "__main__":

    root_dir = r"D:"
    palimpsest_name = "Verona_msXL"
    main_data_dir = os.path.join(root_dir, palimpsest_name)
    ut_folio_name = "msXL_335v_b"
    modality = "M"
    class_name = "undertext"
    n = 3
    box = None
    crop_box = None #[3095, 801, 1362, 453]
    crop_box_str = "" if crop_box is None else "_"+"_".join(crop_box)
    method = "euclidean"
    distances_dir = os.path.join(root_dir,palimpsest_name,ut_folio_name,f"distances_{method}")


    for fpath in [f"msXL_335v_b_msXL_335v_b_{method}_nn_3.pkl"]:
                  #os.listdir(distances_dir):
        if not fpath.endswith(".pkl"):
            continue
        #fpath = os.path.join(distances_dir,ut_folio_name+"_"+folio_name+"_"+f"euclid_nn_{n}.pkl")
        folio_name = "_".join(fpath.split("_")[3:6])
        print(
            f"Processing file: {fpath} for folio: {folio_name}")
        dict_ut = read_pickle(os.path.join(distances_dir,fpath))
        dict_ut = crop_coord(dict_ut,box=crop_box)



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
            output_path = os.path.join(distances_dir, folio_name+f"_coordinates_nn{i}{crop_box_str}.png")
            io.imsave(output_path, colored_image)
            filter_dict_withing_radius(new_dict,xs_ut,ys_ut,1,image_shape)



