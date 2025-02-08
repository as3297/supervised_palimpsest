# read whole page
from read_data import read_msi_image_object
from msi_data_as_array import FullImageFromPILImageCube
from util import read_json, read_split_box_coord
import tensorflow as tf
import numpy as np
from skimage import io
import os
import numpy as np


root_dir = r"d:"
palimpsest_name = "Verona_msXL"
main_data_dir = os.path.join(root_dir, palimpsest_name)
folio_names = ["msXL_335v_b",r"msXL_315v_b","msXL_318r_b","msXL_318v_b","msXL_319r_b","msXL_319v_b",
    "msXL_322r_b","msXL_322v_b","msXL_323r_b","msXL_334r_b",
    "msXL_334v_b","msXL_344r_b","msXL_344v_b",r"msXL_315r_b"]
modality = "M"
trial_name ="20250207-133654"
box = None
main_dir = r"c:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training"

def visualize_predictions():
    for folio_name in folio_names:
        features,im_shape = load_image(main_data_dir,folio_name,modality)
        predictions,saved_model_path = predict(main_dir, palimpsest_name, trial_name,features)
        save_prediction_mages(predictions,im_shape,folio_name,saved_model_path)

def load_image(main_data_dir,folio_name,modality):
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
    saved_model_path = os.path.join(main_dir,palimpsest_name,trial_name)
    batch_size = 256*4
    imported = tf.keras.models.load_model(os.path.join(saved_model_path,"model.keras"))
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

def save_prediction_mages(predictions,im_shape,folio_name,saved_model_path):
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
    visualize_predictions()