import tensorflow as tf
from joblib.numpy_pickle_utils import BUFFER_SIZE
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization, ReLU
from pixel_coord import points_coord_in_bbox

print("TensorFlow version:", tf.__version__)
from util import extend_json
from dataset import load_data_for_training
from datetime import datetime
from pil_image_cube import ImageCubePILobject
from msi_data_as_array import PointsfromMSI_PIL
from util import read_band_list,read_json,extend_json,read_split_box_coord
import os
osp = os.path.join


def model_categorical(nb_features,nb_units_per_layer,nb_layers,dropout_rate,batch_norm=False):
    inputs = keras.Input(shape=(nb_features,), name="counts")
    current_nb_units_per_layer = nb_units_per_layer
    for layer_idx in range(nb_layers):
        if layer_idx==0:
           x = Dense(nb_units_per_layer, activation=None, name="dense_{}".format(layer_idx))(inputs)
        else:
            current_nb_units_per_layer = round(1.3*current_nb_units_per_layer)
            x = Dense(nb_units_per_layer, activation=None, name="dense_{}".format(layer_idx))(x)
        if batch_norm:
            x = BatchNormalization(axis=1,name="batchnorm_{}".format(layer_idx))(x)
        x = ReLU(name="relu_{}".format(layer_idx))(x)
        x = Dropout(dropout_rate,name="dropout_{}".format(layer_idx))(x)
        outputs = Dense(3, name="predictions",activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def build_model(model_par):
    model = model_categorical(**model_par)
    model.compile(loss="categorical_crossentropy",
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def read_split_box_coords(main_dir,folio_name,split_name):
    bbox_fpath = osp(main_dir, folio_name, "dataset_split.json")
    bbox_dict = read_json(bbox_fpath)
    bbox = read_split_box_coord(split_name, bbox_dict)
    return bbox

def read_feature_map_coords(main_dir,folio_name,class_name,split_name):
    fpath_image_mask = os.path.join(main_dir, folio_name, "mask", f"{folio_name}-{class_name}_black.png")
    bbox = read_split_box_coords(main_dir,folio_name,split_name)
    xs, ys, _ = points_coord_in_bbox(fpath_image_mask, bbox)
    return xs, ys

def generator(main_dir,modality,folio,class_names_dict):
    band_list_file = osp(main_dir, "band_list.txt")
    band_list = read_band_list(band_list_file, modality)
    BUFFER_SIZE = 4
    rotate_angle = 0  # Assigning a default rotate angle
    im_pil_ob = ImageCubePILobject(main_dir, folio, band_list, rotate_angle)
    for split_name in ["train", "val", "test"]:
        for class_name, class_idx in class_names_dict.items():
            xs, ys = read_feature_map_coords(main_dir,folio,class_name,split_name)
            points_object = PointsfromMSI_PIL(pil_msi_obj=im_pil_ob, points_coord=list(zip(xs, ys)))
            features = points_object.points
            labels = [class_idx]*len(xs)
            ds = tf.data.Dataset.from_tensor_slices({"xs,ys":(xs,ys),"features":features,"label":labels})
            ds = ds.shuffle(BUFFER_SIZE).repeat()
            


def save_training_parameters(gr,debugging,batch_size,nb_epochs,nb_features,learning_rate_decay_epoch_step,dropout_rate):
  d = {}
  d["restore_path"] = gr.restore_path
  d["debugging"] = debugging
  d["batch_size"] = batch_size
  d["nb_epochs"] = nb_epochs
  d["nb_layers"] = gr.nb_layers
  d["optimizer"] = gr.optimizer.get_config()
  d["learning_rate"] = gr.learning_rate
  d["loss_function"] = gr.loss_object.get_config()
  d["nb_units_per_layer"] = gr.nb_units_per_layer
  d["nb_features"] = nb_features
  d["learning_rate_decay_epoch_step"] = learning_rate_decay_epoch_step
  d["model_dir"] = gr.model_dir
  d["label_smoothing"] = gr.loss_object.label_smoothing
  d["loss"] = gr.loss_object.name
  d["weight_decay"] = gr.optimizer.get_config()["weight_decay"]
  d["dropout_rate"] = dropout_rate
  extend_json(osp(gr.model_dir,"training_parameters.json"),d)


def training(restore_path = None,debugging=False):
    """
    Trains a machine learning model. Optionally, the training process can be continued from a previous checkpoint and debugging information can be enabled.

    Arguments:
    restore_path : str or None
        Path to the previously saved model checkpoint from which to continue training. If None, training starts from scratch.
    debugging : bool
        Flag to enable or disable debugging information during the training process. Default is False.
    """
    EPOCHS = 500
    batch_size = 32*4
    modalities = ["M"]
    nb_nodes_in_layer = 20
    nb_layers = 5
    optimizer_name = "adamw"
    weight_decay = 0.1
    learning_rate = 0.0001
    dropout_rate = 0.5
    label_smoothing = 0.1
    loss_name = "binary_crossentropy"
    main_data_dir = r"C:\Data\PhD\palimpsest\Victor_data"
    palimpsest_dir = r"Paris_Coislin"
    base_data_dir = osp(main_data_dir,palimpsest_dir)
    folios = [r"Par_coislin_393_054r"]
    ut_mask_file = r"knn_undertext_mask"
    nonut_mask_file = r"not_undertext_black"
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(r"C:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training",palimpsest_dir, current_time)
    learning_rate_decay_epoch_step = 0
    class_balansing_factor = 0.175
    #Early stopping parametrs
    patience = 15

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    trainset_ut, trainset_nonut, valset_ut, valset_nonut = load_data_for_training(
      model_dir, modalities,base_data_dir,folios,ut_mask_file,nonut_mask_file, debugging)
    print("Number of train ut samples: ", len(trainset_ut[0]))
    print("Number of train non ut samples: ", len(trainset_nonut[0]))
    model = build_model()
    history = model.fit(
        train_generator,
        epochs=nb_epoch, validation_data=val_generator)
    loss_aver_val = []
    loss_aver_train = []
    loss_aver_val.append(history.history['val_loss'])
    loss_aver_train.append(history.history['loss'])

def testing(saved_model_path):
  imported = tf.saved_model.load(saved_model_path)
  pass

if __name__=="__main__":
  training(None,False)
