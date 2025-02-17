import sys
import os
# Add the root directory (one level up) to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from read_data import read_msi_image_object
from msi_data_as_array import FullImageFromPILImageCube
from util import read_json, read_split_box_coord

import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage import io
import os
import numpy as np

def visualize_predictions(data_dir,folio_names,model_name,main_dir,palimpsest_name,modality,box):
    """
    Run model on the data store the prediction as images

    :param data_dir: Directory containing dataset images
    :param folio_names: List of folio image filenames to be processed
    :param model_name: Name of the machine learning model used for prediction
    :param main_dir: Main directory containing model and results
    :param palimpsest_name: Identifier for the palimpsest dataset
    :param modality: Modality of the image data (e.g., RGB, infrared)
    :param box: Coordinates defining the region of interest in the images
    :return: None
    """
    for folio_name in folio_names:
        features,im_shape = load_image(data_dir,folio_name,modality,box)
        predictions,saved_model_path = predict(main_dir, palimpsest_name, model_name,features)
        save_prediction_mages(predictions,im_shape,folio_name,saved_model_path,box)

def load_image(main_data_dir,folio_name,modality,box):
    """
    :param main_data_dir: Path to the main directory containing the data.
    :param folio_name: Name of the specific dataset or folio.
    :param modality: The modality type used to read the image object.
    :param box: Optional parameter specifying the coordinates of the bounding box in [x_min, y_min, x_max, y_max] format. If provided, restricts image processing to the specified region.
    :return: A tuple containing the reshaped image features and the shape of the processed image.
    """
    im_pil_ob = read_msi_image_object(main_data_dir, folio_name, modality)
    msi_im_obj = FullImageFromPILImageCube(pil_msi_obj=im_pil_ob)
    im_shape = (im_pil_ob.height, im_pil_ob.width)
    msi_im = msi_im_obj.ims_img
    if not box is None:
        bbox_fpath = os.path.join(main_data_dir, folio_name, "dataset_split.json")
        bbox_dict = read_json(bbox_fpath)
        bbox = read_split_box_coord(box, bbox_dict)
        msi_im = msi_im[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        im_shape = (bbox[3] - bbox[1], bbox[2] - bbox[0])

    features = np.reshape(msi_im, newshape=(-1, im_pil_ob.nb_bands))
    return features,im_shape

def predict(main_dir, palimpsest_name, trial_name,features):
    """
    :param main_dir: The main directory where the model files are stored.
    :param palimpsest_name: The name of the specific sub-directory or category of the model.
    :param trial_name: The name of the trial or experiment associated with the model.
    :param features: The input features as a NumPy array used for making predictions.
    :return: A tuple containing the predictions as a NumPy array and the path to the loaded model.
    """
    saved_model_path = os.path.join(main_dir,palimpsest_name,trial_name)
    batch_size = 256*4
    imported = load_model(os.path.join(saved_model_path,"model.keras"))
    print("Finished model loading.")
    nb_samples = features.shape[0]
    predictions = np.zeros((nb_samples,))
    for idx in range(0,nb_samples, batch_size):
        batch = features[idx:idx + batch_size,:]
        batch = tf.constant(batch,dtype=tf.float32)
        output = imported(batch)
        output = output.numpy()
        predictions[idx:idx + batch_size] = output[:,0]
    return predictions,saved_model_path

def save_prediction_mages(predictions,im_shape,folio_name,saved_model_path,box):
    """
    :param predictions: The prediction results as a NumPy array, typically representing image data.
    :param im_shape: A tuple representing the shape of the image (height, width).
    :param folio_name: A string representing the name or identifier of the current item being processed.
    :param saved_model_path: The file path to the directory where the prediction images will be saved.
    :param box: An identifier or label for the specific image or region being processed.
    :return: None. The function saves prediction images to disk.
    """
    predictions = np.reshape(predictions,newshape=im_shape+(1,))
    predictions_thresh = predictions>0.5
    predictions = (predictions*255).astype(np.uint8)
    predictions_thresh = predictions_thresh.astype(np.uint8)*255
    predictions = np.repeat(predictions,repeats=3,axis=2)
    save_path = os.path.join(saved_model_path,f"{folio_name}_{modality}_prediction_{box}.tif")
    io.imsave(save_path,predictions)
    save_path = os.path.join(saved_model_path, f"{folio_name}_{modality}_prediction_thresh_{box}.tif")
    io.imsave(save_path, predictions_thresh)

if __name__ == "__main__":
    root_dir = r"/projects/palimpsests"#r"d:"
    palimpsest_name = "Verona_msXL"
    main_dir = r"/projects/supervised_palimpsest/training"#r"c:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training"
    model_name = "20250210-235448"
    modality = "M"
    box = None

    folio_names = ["msXL_335v_b", r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
                   "msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b",
                   "msXL_334v_b", "msXL_344r_b", "msXL_344v_b", ]  # r"msXL_315r_b"]

    data_dir = os.path.join(root_dir, palimpsest_name)
    visualize_predictions(data_dir,folio_names,model_name,main_dir,palimpsest_name,modality,box)