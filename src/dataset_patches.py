import copy

import matplotlib.pyplot as plt
import os

import numpy as np
import tensorflow as tf
from PIL import ImageOps
from PIL import Image
from src.pil_image_cube import ImageCubeObject
from src.read_data import read_x_y_coords



class PILMSIPatchExtractor():
    WINDOW_MULTIPLIER = 2
    IMAGE_PADDING_FILL = 0  # Padding fill value used in `read_band_fragment`

    def __init__(self, window_size):
        self.half_window = window_size // 2  # Ensures pixels are centered in the extracted window

    def calculate_window_size(self):
        """Calculate the patch window size."""
        return self.half_window * self.WINDOW_MULTIPLIER + 1

    def extract_coordinates_and_labels(self, class_dict, im_pil_ob,box):
        """Extract point coordinates and their labels from binary masks."""
        points, labels = [], []
        for class_name,label in class_dict.items():
            xs,ys = read_x_y_coords(im_pil_ob.folio_dir,im_pil_ob.folio_name,class_name,box)
            points.extend(zip(xs,ys))
            labels.extend([label]*len(xs))
        return points, labels

    def _create_dataset(self, total_points, batch_size, map_function, prefetch=True, shuffle=False, buffer_size=10000):
        """Create and prepare a tf.data.Dataset."""
        dataset = tf.data.Dataset.from_tensor_slices(np.arange(total_points))
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        # Dynamically setting the output type based on the map function
        is_with_labels = 'with_label' in map_function.__name__  # Ensure map function naming follows convention

        dataset = dataset.map(
            lambda idx: tf.py_function(
                func=map_function,
                inp=[idx],
                Tout=(tf.float32, tf.int32) if is_with_labels else tf.float32
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(batch_size)
        if prefetch:
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def _extract_patch(self, coord, msi_image, window_size):
        """Extract patch for the coordinate from multispectral image."""
        patch = np.zeros([window_size, window_size, msi_image.nb_bands], dtype=np.float32)
        for band_idx,band_name in enumerate(msi_image.band_list):
            patch[:, :, band_idx] = self.read_band_fragment(coord,msi_image,band_name)
        return patch


    def tf_data_from_multiple_folios(self, pil_msi_objs,class_dict, batch_size=32, shuffle=True, buffer_size=10000, box=None):
        """
        Create a combined tf.data.Dataset from multiple pil_msi_obj objects and their respective masks.
        Creates a tf.data.Dataset from binary masks and a multispectral image cube.

        Parameters:
        - pil_msi_objs: list of ImageCubePILobject
            A list of  PIL-based multispectral image objects containing image data.
        - class_dict: dict of "class_name":"label" pairs
        - batch_size: int, optional
            Number of samples in each batch (default: 32).
        - shuffle: bool, optional
            Whether to shuffle the data (default: True).
        - buffer_size: int, optional
            Buffer size for shuffling (default: 10000).

        Example usage for training:

        pil_msi_obj = ...  # Your ImageCubePILobject
        class_dict = {"undertertext": 1, "not_undertext": 0}

        # Initialize the PatchesfromMSI_PIL instance
        patches_instance = PatchesfromMSI_PIL(pil_msi_obj, points_coord=None, win=16)

        # Generate the tf.data.Dataset
        batch_size = 32
        dataset = patches_instance.tf_data_from_multiple_folios(pil_msi_objs, class_dict, batch_size=batch_size, shuffle=True)

        # Train the model
        model.fit(
            dataset,
            epochs=10,
            steps_per_epoch=(sum(mask.sum() for mask in masks) + batch_size - 1) // batch_size
        )

        Returns:
        - tf.data.Dataset
            A TensorFlow dataset yielding batches of patches and labels.
        """

        all_points = []
        all_labels = []
        window_size = self.calculate_window_size()

        # Collect points and labels from all pil_msi_obj objects and masks
        for pil_msi_obj in pil_msi_objs:
            points, labels = self.extract_coordinates_and_labels(class_dict,pil_msi_obj,box)
            all_points.extend([(point, pil_msi_obj) for point in points])  # Keep track of the corresponding msi_image
            all_labels.extend(labels)

        total_points = len(all_points)
        indexes = np.arange(total_points)
        np.random.shuffle(indexes)
        all_points = [all_points[i] for i in indexes]
        all_labels = [all_labels[i] for i in indexes]
        def map_with_labels(idx):
            point, msi_image = all_points[idx]
            return self._extract_patch(point, msi_image, window_size), all_labels[idx]

        return self._create_dataset(total_points, batch_size, map_with_labels, shuffle=shuffle, buffer_size=buffer_size)

    def generate_points_coords(self, pil_msi_obj) -> list[tuple[int, int]]:
        """Generates a list of (x, y) coordinates for all pixels in the given image."""
        return [(x, y) for x in range(pil_msi_obj.width) for y in range(pil_msi_obj.height)]


    def tf_data_for_predictions(self, pil_msi_obj,coords, batch_size=32):
        """
        Creates a tf.data.Dataset for prediction, yielding only patches.

        Parameters:
        - pil_msi_obj: ImageCubePILobject
            A PIL-based multispectral image object containing image data.
        - points_coord: list of tuples
            A list of coordinate points (x, y) to extract patches from.
        - batch_size: int, optional
            Number of samples in each batch (default: 32).
        # Example usage for prediction

        pil_msi_obj = ...  # Your ImageCubePILobject
        points_coord = ...  # List of point coordinates for prediction

        # Create the prediction dataset
        batch_size = 32
        dataset = patches_instance.tf_data_for_prediction(pil_msi_obj, points_coord, batch_size)

        # Make predictions
        predictions = model.predict(dataset, steps=(len(points_coord) + batch_size - 1) // batch_size)

        Returns:
        - tf.data.Dataset
            A TensorFlow dataset yielding batches of patches.
        """
        window_size = self.calculate_window_size()


        if len(coords)==0:
            coords = self.generate_points_coords(pil_msi_obj)
        all_points = [(point, pil_msi_obj) for point in coords]

        total_points = len(all_points)

        def map_without_labels(idx):
            point, msi_image = all_points[idx]
            return self._extract_patch(point, msi_image, window_size)

        return self._create_dataset(total_points, batch_size, map_without_labels, prefetch=True, shuffle=False)

    def read_band_image(self,pil_msi_obj,band_name):

        fpath = os.path.join(pil_msi_obj.image_dir, pil_msi_obj.folio_name + "-" + band_name + ".tif")
        if not os.path.exists(fpath):
            fpath = os.path.join(pil_msi_obj.image_dir, pil_msi_obj.folio_name + "+" + band_name + ".tif")
        image_band = Image.open(fpath)
        if pil_msi_obj.rotate_angle > 0:
            rotation = eval("Image.ROTATE_{}".format(pil_msi_obj.rotate_angle))
            image_band = image_band.transpose(rotation)
        return image_band

    def read_band_fragment(self, coord, pil_msi_obj,band_name):
        """Read a fragment of an image band at the specified coordinate."""
        image_band = self.read_band_image(pil_msi_obj,band_name)
        img_width, img_height = image_band.size
        left = max(0, coord[0] - self.half_window)
        upper = max(0, coord[1] - self.half_window)
        right = min(img_width, coord[0] + self.half_window + 1)
        lower = min(img_height, coord[1] + self.half_window + 1)
        if left >= right or upper >= lower:
            raise ValueError(
                f"Invalid crop coordinates: ({left}, {upper}) exceed image size ({img_width}, {img_height})")
        fragment = image_band.crop((left, upper, right, lower))
        pad_left, pad_top = max(0, self.half_window - coord[0]), max(0, self.half_window - coord[1])
        pad_right, pad_bottom = max(0, (coord[0] + self.half_window + 1) - img_width), max(0, (
                    coord[1] + self.half_window + 1) - img_height)
        if any((pad_left, pad_top, pad_right, pad_bottom)):
            fragment = ImageOps.expand(fragment, border=(pad_left, pad_top, pad_right, pad_bottom),
                                       fill=self.IMAGE_PADDING_FILL)
        fragment = np.array(fragment)
        image_band.close()
        return fragment

def dataset_tf(main_dir,folio_names,modalities,window_size,batch_size,class_dict,shuffle=True,buffer_size=10000,box=None):
    pil_msi_objs = []
    for folio_name in folio_names:
        pil_msi_obj = ImageCubeObject(main_data_dir, folio_name, modalities, 0)
        pil_msi_objs.append(pil_msi_obj)
    patches_instance = PILMSIPatchExtractor(window_size)
    dataset = patches_instance.tf_data_from_multiple_folios(pil_msi_objs, {"undertext": 1}, batch_size=batch_size,shuffle=shuffle,buffer_size=buffer_size,box=box)
    return dataset

if __name__ == "__main__":
    root_dir = r"D:"
    palimpsest_name = "Verona_msXL"
    main_data_dir = os.path.join(root_dir, palimpsest_name)
    folio_names = ["msXL_335v_b", "msXL_319v_b"]
    modalities = ["M"]
    pil_msi_objs = []
    for folio_name in folio_names:
        pil_msi_obj = ImageCubeObject(main_data_dir, folio_name, modalities, 0)
        pil_msi_objs.append(pil_msi_obj)
    patches_instance = PILMSIPatchExtractor(2)
    points_coord = [(1200, 2200)]
    #dataset = patches_instance.tf_data_for_predictions(pil_msi_objs, points_coord)
    dataset = patches_instance.tf_data_from_multiple_folios([pil_msi_obj], {"undertext": 1,"not_undertext":0}, shuffle=True,batch_size=2)
    for batch,labels in dataset:
        print(batch.shape)
        print(labels)
    plt.figure()
    plt.imshow(batch[0,:,:,0])
    plt.show()
