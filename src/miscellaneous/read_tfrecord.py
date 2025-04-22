import matplotlib.pyplot as plt
from tensorflow.data import TFRecordDataset
import tensorflow as tf


def read_tfrecord_file(file_path, feature_description):
    """
    Reads and parses a TFRecord file.
    
    Args:
        file_path: Path to the TFRecord file.
        feature_description: A dictionary describing the features stored in the TFRecord
                             file for parsing.
    
    Returns:
        A parsed dataset as a tf.data.Dataset object.
    """
    raw_dataset = TFRecordDataset(file_path)

    # Parse the data from TFRecords
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset
feature_description = {
        'band_list': tf.io.FixedLenFeature([22], tf.string),  # Example band list, adjust as needed
        'palimpsest_name': tf.io.FixedLenFeature([], tf.string),
        'folio_name': tf.io.FixedLenFeature([], tf.string),
        'coords': tf.io.FixedLenFeature([2], tf.int64), # Store the coordinates as a list,
        'patch_raw': tf.io.FixedLenFeature([], tf.string), # Store the serialized tensor bytes
        'label': tf.io.FixedLenFeature([], tf.int64),          # Store the integer label
    }

parsed_dataset = read_tfrecord_file(r"D:\Verona_msXL\msXL_335v_b\undertext_11.tfrecord", feature_description)
# Decode image from serialized bytes
for record in parsed_dataset:
    # Access the 'patch_raw' field and decode it
    patch_raw = record['patch_raw']  # Still a serialized string tensor
    patch= tf.io.parse_tensor(patch_raw, out_type=tf.float32)  # Decode the serialized tensor
    #patch = tf.io.decode_raw(patch_raw, tf.uint16)  # Example dtype: adjust as needed
    coords = (record['coords']).numpy()
    label = record['label'].numpy()
    # Reshape or interpret the patch if necessary
    # Assuming it's a 2D or 3D representation (update shape accordingly)
    height, width, channels = 11, 11, 22  # Adjust based on your data's actual shape
    patch = tf.reshape(patch, (height, width, channels))

    # Visualize the patch
    plt.figure()
    plt.imshow(patch[:, :, 0], cmap='gray')
    plt.show()
    break  # Remove the break to process all records
