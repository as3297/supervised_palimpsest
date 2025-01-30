from random import shuffle

import numpy as np
import tensorflow as tf
from read_data import read_msi_image_object, read_subset_features
from msi_data_as_array import PointfromMSI_PIL,PointsfromMSI_PIL


def create_tf_record(main_dir,folio_name,modality,class_names,labels_dict):
    """Creates TF Record file with stored multispectral values of selected pixels"""
    # Sample data
    for class_name in class_names:
        label = 1
    features_nonut, xs_nonut, ys_nonut = read_subset_features(main_dir, folio_name, class_name, modality, box=None)

    # Create a tf.train.Example
    example = tf.train.Example(features=tf.train.Features(feature={
        'feautures': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))

    # Write to TFRecord
    with tf.io.TFRecordWriter('my_data.tfrecord') as writer:
        writer.write(example.SerializeToString())


#TODO: test the actual values from dataset
def create_oversampled_dataset_test():
    batch_size = 256
    main_dir = r"D:\Verona_msXL"
    pal_names = ["msXL_315r_b","msXL_319r_b"]
    modality = "M"
    resample_ds = create_oversampled_dataset(batch_size,pal_names,main_dir,modality,False,False,True)
    print(
        f"The dataset contains {resample_ds.cardinality().numpy()} elements, with {resample_ds.element_spec} as element spec."
    )
    for element in resample_ds:
        print(element)
        pixel_values = element[0].numpy()
        coords = element[2].numpy()
        pal_names = element[4].numpy()
        for pixel_value, coord, pal_name in zip(pixel_values, coords, pal_names):
            im_pil_ob = read_msi_image_object(main_dir, pal_name.decode("utf-8"), modality)
            points_object = PointfromMSI_PIL(pil_msi_obj=im_pil_ob, point_coord= list(coord))
            features = points_object.point
            print(pixel_value)
            print(features)
            if np.any(pixel_value != features):
                AssertionError("Pixel values do not match")


if __name__ == "__main__":
    create_oversampled_dataset_test()