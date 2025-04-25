import tensorflow as tf
import os
from matplotlib import pyplot as plt


def standartize(im, spectralon_mean):
    """Standartized image by grey_value
    im  - is tensor of shape (height, width, channels)
    spectralon_mean - is tensor of shape (channels,)"""

    im = im / tf.math.maximum(spectralon_mean, 1e-6)
    im = tf.clip_by_value(im, clip_value_min=0.0, clip_value_max=1.0)

    return im

# Parse the TFRecord Example
def parse_tfrecord_fn(example):
    """
    Parse the serialized TFRecord Example.
    Adjust the features dictionary to match your TFRecord structure.
    """
    feature_description = {
        'band_list': tf.io.VarLenFeature(tf.string),
        'palimpsest_name': tf.io.FixedLenFeature([], tf.string),
        'folio_name': tf.io.FixedLenFeature([], tf.string),
        'coords': tf.io.FixedLenFeature([2], tf.int64),
        'patch_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'patch_shape': tf.io.FixedLenFeature([3], tf.int64),
        'spectralon_mean': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)

    # Deserialize tensors if needed
    parsed_example['patch_raw'] = tf.io.parse_tensor(parsed_example['patch_raw'], out_type=tf.uint16)
    parsed_example['spectralon_mean'] = tf.io.parse_tensor(parsed_example['spectralon_mean'], out_type=tf.float32)
    # Reshape the patch tensor
    patch_shape = parsed_example['patch_shape']
    patch_shape = tf.cast(patch_shape, tf.int32)
    parsed_example['patch_raw'] = tf.reshape(parsed_example['patch_raw'], patch_shape)
    # Convert patch to float32
    parsed_example['patch_raw'] = tf.cast(parsed_example['patch_raw'], tf.float32)
    # Scale patch by spectralon mean
    out = standartize(parsed_example['patch_raw'], parsed_example['spectralon_mean'])

    return out, parsed_example['label']

# --- Building a tf.data Dataset that avoids tf.py_function ---

def dataset_tf(main_data_dir, folio_names,window_size,classes_dict, batch_size):
    """
    Create a tf.data pipeline to read, parse, shuffle, and batch samples from TFRecord files.
    Args:
        tfrecord_dir: Directory where TFRecord files are stored.
        batch_size: Batch size for training.
    Returns:
        A tf.data.Dataset object.
    """
    tfrecord_files = []
    for folio_name in folio_names:
        for class_name, label in classes_dict.items():
            tfrecord_files.append(
                os.path.join(main_data_dir, folio_name, f"{class_name}_{window_size}.tfrecords"))


    # Shuffle the list of files
    tfrecord_files = tf.random.shuffle(tf.convert_to_tensor(tfrecord_files))

    # Use interleave to read multiple files concurrently for better performance
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    dataset = dataset.interleave(
        lambda file: tf.data.TFRecordDataset(file).map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE),
        cycle_length=tf.data.AUTOTUNE,  # Number of files to read in parallel
        block_length=1,  # How many records from each file should we read at a time
        num_parallel_calls=tf.data.AUTOTUNE  # Processing in parallel
    )

    # Shuffle, batch, and prefetch for performance
    dataset = dataset.shuffle(buffer_size=1000)  # Adjust buffer_size based on data size
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset





if __name__ == "__main__":

    main_data_dir = r"D:\Verona_msXL"
    folio_names = [r"msXL_335v_b",] #"msXL_319v_b"]
    modalities = ["M"]
    class_dict = {"undertext":1,"not_undertext":0}
    window_size = 11

    dataset = dataset_tf(main_data_dir,folio_names,window_size,class_dict, 32)

    # Iterate over one batch.
    for patches, labels in dataset.take(1):
        print("Batch patches shape:", patches.shape)
        print("Batch labels:", labels.numpy())
        for i in range(patches.shape[0]):
            print(f"Patch {i} shape:", patches[i].shape)
            print(f"Label {i}:", labels[i].numpy())
            # Visualize the first channel of the first patch
            plt.figure()
            plt.imshow(patches[i, :, :, 0], cmap='gray')
            plt.title(f"Patch {i} - Label: {labels[i].numpy()}")
plt.show()






