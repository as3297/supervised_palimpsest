import os

import numpy as np
from matplotlib import pyplot as plt

from src.dataset_patches import ImageCubeObject
import tensorflow as tf

from src.read_data import read_x_y_coords
from src.util import read_band_list


# --- Utility functions implemented with pure TensorFlow ops ---

def read_png_image_tf(file_path):
    """
    Reads a png image file, decodes it, and optionally rotates it.
    Args:
      file_path: A scalar string tensor with the image file path.
      rotate_angle: An integer; must be 0, 90, 180, or 270.
    Returns:
      A decoded image tensor with shape [H, W, C].
    """
    image_bytes = tf.io.read_file(file_path)
    # Decode using TensorFlow I/O (assumes the image is a TIFF)
    image = tf.image.decode_png(image_bytes)
    return image


def extract_patch_tf(image, coord, half_window, window_size, padding_fill=0):
    """
    Extracts a patch of size [window_size, window_size] from the image tensor.
    Args:
      image: A tensor of shape [H, W, C].
      coord: A tensor [x, y] representing the center pixel.
      half_window: Integer, equal to window_size // 2.
      window_size: Desired final patch size (square).
      padding_fill: Padding value for regions outside the image.
    Returns:
      A tensor of shape [window_size, window_size, C].
    """
    shape = tf.shape(image)
    H, W = shape[0], shape[1]
    x, y = coord[0], coord[1]

    # Calculate crop boundaries
    left = tf.maximum(0, x - half_window)
    upper = tf.maximum(0, y - half_window)
    right = tf.minimum(W, x + half_window + 1)
    lower = tf.minimum(H, y + half_window + 1)

    cropped = tf.image.crop_to_bounding_box(
        image,
        offset_height=upper,
        offset_width=left,
        target_height=lower - upper,
        target_width=right - left
    )

    # Compute padding amounts
    pad_top = tf.maximum(0, half_window - y)
    pad_left = tf.maximum(0, half_window - x)
    pad_bottom = tf.maximum(0, (x + half_window + 1) - W)
    pad_right = tf.maximum(0, (y + half_window + 1) - H)
    paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

    cropped = tf.pad(cropped, paddings, constant_values=padding_fill)
    patch = tf.image.resize_with_crop_or_pad(cropped, window_size, window_size)
    patch = tf.cast(patch, tf.float32) / 255.0

    return patch


def process_patch_with_label(patch_spec, window_size, band_list,padding_fill):
    """
    Process a patch specification to extract a patch from each band and return the patch with label.

    Args:
      patch_spec: A tuple (base_path, coord, label) where:
          - base_path is a scalar string tensor (e.g. '/data/folio1/MS_XL_315r')
          - coord is a tensor [x, y]
          - label is a scalar (e.g. 0 or 1)
      window_size: The final patch size (square).
      band_list: List of band names (strings).
      rotate_angle: Rotation angle in degrees (e.g., 0, 90, 180, 270).
      padding_fill: Padding fill value.

    Returns:
      A tuple: (patch, label), where patch is a tensor of shape [window_size, window_size, num_bands].
    """
    half_window = window_size // 2
    base_path, coord, label = patch_spec

    patches = []
    # Loop over each band and build the file path. Here we assume your file naming follows:
    # "<base_path>-<band>.tif"
    for band in band_list:
        file_path = tf.strings.join([base_path, "-", band, ".png"])
        image = read_png_image_tf(file_path)
        patch = extract_patch_tf(image, coord, half_window, window_size, padding_fill)
        patch = tf.squeeze(patch)
        patches.append(patch)
    # Stack patches along the channel axis.

    patch_tensor = tf.stack(patches, axis=-1)
    return patch_tensor, label


# --- Building a tf.data Dataset that avoids tf.py_function ---

def build_patch_dataset_with_labels(patch_specs, window_size, band_list, padding_fill=0, batch_size=32,
                                    shuffle=True, buffer_size=10000):
    """
    Builds a tf.data.Dataset yielding patches and labels.
    Each element of patch_specs is a tuple (base_path, coord, label).

    Args:
      patch_specs: List of tuples (base_path, [x, y], label).
      window_size: Integer patch size.
      band_list: List of band names.
      rotate_angle: Rotation angle.
      padding_fill: Value for padding.
      batch_size: Batch size.
      shuffle: Whether to shuffle.
      buffer_size: Buffer size.

    Returns:
      A tf.data.Dataset yielding (patch, label) pairs.
    """
    # Unpack patch_specs into three lists.
    base_paths, coords, labels = zip(*patch_specs)

    base_paths_tensor = tf.constant(base_paths)
    coords_tensor = tf.constant(coords, dtype=tf.int32)  # shape [N, 2]
    labels_tensor = tf.constant(labels)  # shape [N]

    ds = tf.data.Dataset.from_tensor_slices((base_paths_tensor, coords_tensor, labels_tensor))

    if shuffle:
        ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda base_path, coord, label: process_patch_with_label(
            (base_path, coord, label), window_size, band_list, padding_fill),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def test_build_patch_dataset_with_labels(root_dir,palimpsest_name,main_data_dir,folio_names,modalities,):
    coords = [(100, 150), (200, 180), (50, 75)]
    labels = [1, 0, 1]
    patch_specs = []
    for i, folio_name in enumerate(folio_names):
        folio_dir = os.path.join(root_dir, palimpsest_name, "png_images_standardized", folio_name, folio_name)
        patch_specs.append((folio_dir, coords[i], labels[i]))
    band_list = read_band_list(os.path.join(main_data_dir, "band_list.txt"), modalities)

    window_size = 33  # For example, a 33x33 patch.
    rotate_angle = 0  # Rotation in degrees.
    padding_fill = 0
    batch_size = 32

    dataset = build_patch_dataset_with_labels(patch_specs, window_size, band_list, rotate_angle, padding_fill,
                                              batch_size)


def dataset_tf(main_data_dir,folio_names,classes_dict,modalities,window_size, rotate_angle, batch_size=32, shuffle=True,
                        buffer_size=10000,box=None):
    png_folder = "png_images_standardized"
    patch_specs = []
    for folio_name in folio_names:
        msi_obj = ImageCubeObject(main_data_dir, folio_name, modalities, 0)
        for class_name,label in classes_dict.items():
            xs, ys = read_x_y_coords(msi_obj.folio_dir, msi_obj.folio_name, class_name, box)
            coords = zip(xs, ys)
            for coord in coords:
                patch_specs.append((os.path.join(main_data_dir,png_folder,folio_name,folio_name),coord,label))

    total_points = len(patch_specs)
    indexes = np.arange(total_points)
    np.random.shuffle(indexes)
    patch_specs = [patch_specs[i] for i in indexes]
    band_list = read_band_list(os.path.join(main_data_dir, "band_list.txt"), modalities)
    dataset = build_patch_dataset_with_labels(patch_specs, window_size, band_list, 0, batch_size, shuffle, buffer_size)
    return dataset


if __name__ == "__main__":
    root_dir = r"D:"
    palimpsest_name = "Verona_msXL"
    main_data_dir = os.path.join(root_dir,palimpsest_name)
    folio_names = ["msXL_335v_b", "msXL_335v_b","msXL_319v_b"]
    modalities = ["M"]
    class_dict = {"undertext":1,"not_undertext":0}
    dataset = dataset_tf(main_data_dir,folio_names,class_dict,modalities,window_size=33,rotate_angle=0,batch_size=32,shuffle=True,buffer_size=10000)
    # Iterate over one batch.
    for patches, labels in dataset.take(1):
        print("Batch patches shape:", patches.shape)
        print("Batch labels:", labels.numpy())
    plt.figure()
    plt.imshow(patches[0, :, :, 0])
    plt.show()






