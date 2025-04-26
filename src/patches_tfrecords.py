import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm  # For progress bar (install with: pip install tqdm)
from msi_data_as_array import PatchesfromMSI_PIL
from pil_image_cube import ImageCubePILobject
from read_data import read_x_y_coords
import time

from src.util import read_band_list
from tests.test_dataset import palimpsest_name


# --- TFRecord Helper Functions ---

def _bytes_feature(value):
    """Returns a bytes_list from a string, byte, or list of such values."""

    # If the value is an eager tensor, convert it to a NumPy array then to bytes.
    if tf.is_tensor(value):
        value = value.numpy()

    # Ensure all elements are bytes
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):  # If all elements are strings
            value = [v.encode('utf-8') for v in value]
        elif not all(isinstance(v, bytes) for v in value):  # Mixed or unsupported types in list
            raise TypeError(f"Unsupported type(s) in list for bytes feature: {type(value)}")
    else:  # Single value case
        if isinstance(value, str):
            value = value.encode('utf-8')  # Encode single string to bytes
        elif not isinstance(value, bytes):
            raise TypeError(
                f"Unsupported type for bytes feature: {type(value)}. Expected `str`, `bytes`, list, or Tensor-like types."
            )
        value = [value]  # Wrap single bytes/encoded string into a list

    # Ensure the final value is a list of bytes
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  # Ensure value is a list of floats
  if not isinstance(value, list):
      value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  # Ensure value is a list of integers
  if not isinstance(value, list):
      value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_tf_example(patch_tensor, label, palimpsest_name, folio_name,coords,band_list,spectralon_mean):
    """
    Creates a tf.train.Example proto for a multi-band patch and its label.
    Serializes the patch tensor using tf.io.serialize_tensor.

    Args:
        patch_tensor: The final patch Tensor [H, W, D], dtype=float32.
        label: The integer label for the patch.

    Returns:
        A tf.train.Example proto.
    """
    patch_shape = list(patch_tensor.shape)
    #serealize spectralon mean
    spectralon_mean = tf.io.serialize_tensor(spectralon_mean)

    # Serialize the tensor to bytes
    patch_bytes = tf.io.serialize_tensor(patch_tensor)
    # Encode each string in the list into UTF-8
    encoded_band_list = [s.encode('utf-8') for s in band_list]

    # Create the feature dictionary
    feature = {
        'band_list': _bytes_feature(encoded_band_list),  # Example band list, adjust as needed
        'palimpsest_name': _bytes_feature(palimpsest_name.encode('utf-8')),
        'folio_name': _bytes_feature(folio_name.encode('utf-8')),
        'coords': _int64_feature([coords[0], coords[1]]), # Store the coordinates as a list,
        'patch_raw': _bytes_feature(patch_bytes), # Store the serialized tensor bytes
        'label': _int64_feature(label), # Store the integer label
        'patch_shape': _int64_feature(patch_shape), # Store the shape of the patch
        'spectralon_mean': _bytes_feature(spectralon_mean), # Store the spectralon mean
    }

    # Create and return the Example proto
    return tf.train.Example(features=tf.train.Features(feature=feature))

# --- Main Preprocessing Function ---

def create_tfrecords(base_data_dir,folio_names,window_size, classes_dict,modalities):
    """
    Generates TFRecord file(s) containing pre-extracted multi-band patches by
    iterating through patch specifications.

    Args:
        patch_specs: List of tuples (base_path, coord, label).
                     base_path (str): Path prefix for band files (e.g., '/path/to/folio1/folio1').
                     coord (list/tuple): [x, y] coordinates (integers).
                     label (int): Class label.
        band_list: List of band name strings (e.g., ['band1', 'band2']).
        window_size: Integer size of the square patches (e.g., 11).
        output_tfrecord_path: Path to save the output TFRecord file.
        padding_fill: Value used for padding outside image boundaries.
    """

    start_time = time.time()
    count = 0
    skipped_count = 0
    BOX = None
    chunk_size = 10000
    # Use tf.io.TFRecordWriter within a 'with' statement for safe file handling
    band_list = read_band_list(os.path.join(main_data_dir, "band_list.txt"), modalities)

    for folio_name in folio_names:
        for class_name, label in classes_dict.items():
            # Iterate through patch specifications with a progress bar
            palimpsest_name = os.path.basename(base_data_dir)
            xs,ys = read_x_y_coords(base_data_dir, folio_name, class_name, BOX)

            coords = list(zip(xs, ys))
            output_tfrecord_path = os.path.join(base_data_dir, folio_name, f"{class_name}_{window_size}.tfrecords")
            print(f"Starting TFRecord creation: {output_tfrecord_path}")

            #try:
            with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
                for coords_idx in tqdm(range(0,len(coords),chunk_size), desc=f"Processing Coordinates of {folio_name} class {class_name}"):
                    coords_chunk = coords[coords_idx:min(coords_idx+chunk_size,len(coords))]
                    pil_msi_obj = ImageCubePILobject(base_data_dir, folio_name, band_list, 0)
                    patchs_obj = PatchesfromMSI_PIL(pil_msi_obj,coords_chunk,window_size)
                    patchs = patchs_obj.unstretch_ims_imgs
                    # reshape patches to (BATCH,H, W, BANDS)
                    patchs = patchs.transpose(1,2,3,0)
                    spectralon_mean = patchs_obj.max_vals
                    patch_specs = zip(patchs, coords_chunk)

                    for patch, coord in patch_specs:
                        # Create the tf.train.Example proto containing the patch and label
                        tf_example = create_tf_example(patch, label, palimpsest_name, folio_name,coord,band_list,spectralon_mean)

                        # Write the serialized proto to the TFRecord file
                        writer.write(tf_example.SerializeToString())
                        count += 1
            #except Exception as e:
            #    print(f"\nFATAL ERROR during TFRecord writing: {e}")
            #    # Handle fatal errors, maybe clean up partial file if necessary
            #finally:
            #    end_time = time.time()
            #    duration = end_time - start_time
            #    print(f"\n--- TFRecord Creation Summary ---")
            #    print(f"Output file: {output_tfrecord_path}")
            #    print(f"Total patches written: {count}")
            #    print(f"Total patches skipped due to errors: {skipped_count}")
            #    print(f"Duration: {duration:.2f} seconds")
            #    print(f"---------------------------------")


if __name__ == "__main__":
    main_data_dir = "/projects/palimpsests/Verona_msXL"
    modality = "M"
    folio_names = ["msXL_323r_b", "msXL_334r_b",
                   "msXL_334v_b", "msXL_344r_b", "msXL_344v_b", r"msXL_315r_b"]
    #["msXL_335v_b", r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
    #"msXL_322r_b", "msXL_322v_b",]
    classes_dict = {"undertext": 1, "not_undertext": 0}
    window_size = 11

    create_tfrecords(main_data_dir,folio_names,window_size,classes_dict,modality)
