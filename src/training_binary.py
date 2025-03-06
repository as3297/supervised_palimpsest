import numpy as np
import tensorflow as tf
from model import build_model
from util import extend_json, save_json, convert_float_in_dict
from datetime import datetime
import os
from dataset import dataset

osp = os.path.join


class PalGraph():
  def __init__(self,nb_features,nb_units_per_layer,model_dir,nb_layers,restore_path,
               optimizer_name,label_smoothing,loss,dropout_rate,learning_rate):
    # Create an instance of the model
    self.nb_units_per_layer = nb_units_per_layer
    self.nb_layers = nb_layers
    self.restore_path = restore_path
    self.learning_rate = learning_rate
    if loss == "binary_crossentropy":
      self.loss_object = tf.keras.losses.BinaryCrossentropy(
          from_logits=False,
          label_smoothing=label_smoothing,
          reduction='sum_over_batch_size',
          name='binary_crossentropy'
      )

    if optimizer_name == "adam":
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    elif optimizer_name == "sgd":
        self.optimizer = tf.keras.optimizers.SGD(learning_rate= self.learning_rate)

    if restore_path is None:
          self.model = build_model(nb_features,nb_units_per_layer, nb_layers, dropout_rate)
          self.model_dir = model_dir
          self.model.compile(
        optimizer=self.optimizer,
        loss=self.loss_object,
        loss_weights=None,
        metrics=["accuracy"],
)
    else:
          self.model_dir = restore_path
          # Load the optimizer weights
          opt_weights = np.load(osp(self.model_dir,'optimizer.npy'), allow_pickle=True)
          grad_vars = self.model.trainable_weights
          # This need not be model.trainable_weights; it must be a correctly-ordered list of
          # grad_vars corresponding to how you usually call the optimizer.
          zero_grads = [tf.zeros_like(w) for w in grad_vars]
          # Apply gradients which don't do nothing with Adam
          self.optimizer.apply_gradients(zip(zero_grads, grad_vars))
          # Set the weights of the optimizer
          self.optimizer.set_weights(opt_weights)
          self.model = tf.keras.models.load_model(os.path.join(restore_path, 'model.keras'))


def save_training_parameters(gr,debugging,batch_size,nb_epochs,nb_features,learning_rate_decay_epoch_step,
                             dropout_rate,label_smoothing,weight_decay,patience):
  d = {}
  d["restore_path"] = gr.restore_path
  d["debugging"] = debugging
  d["batch_size"] = int(batch_size)
  d["nb_epochs"] = int(nb_epochs)
  d["nb_layers"] = int(gr.nb_layers)
  d["optimizer"] = convert_float_in_dict(gr.optimizer.get_config())
  d["learning_rate"] = float(gr.learning_rate)
  d["loss_function"] = convert_float_in_dict(gr.loss_object.get_config())
  d["nb_units_per_layer"] = int(gr.nb_units_per_layer)
  d["nb_features"] = int(nb_features)
  d["learning_rate_decay_epoch_step"] = int(learning_rate_decay_epoch_step)
  d["model_dir"] = gr.model_dir
  d["label_smoothing"] = float(label_smoothing)
  d["loss"] = gr.loss_object.name
  d["weight_decay"] = float(weight_decay)
  d["dropout_rate"] = float(dropout_rate)
  d["patience"] = int(patience)
  save_path = osp(gr.model_dir,"training_parameters.json")
  if not os.path.exists(save_path):
     save_json(save_path, d)
  else:
    extend_json(save_path, d)

def save_dataset_par(train_folios,val_folios,model_dir,classes_dict):
    d = {}
    d["train_folios"] = train_folios
    d["val_folios"] = val_folios
    d["classes_dict"] = classes_dict
    save_path = osp(model_dir, "dataset_parameters.json")
    if not os.path.exists(save_path):
        save_json(save_path, d)
    else:
        extend_json(save_path, d)



def training(
            current_time=datetime.now().strftime("%Y%m%d-%H%M%S"),
            model_dir=r"/projects/supervised_palimpsest/training",
            epochs=500,
            batch_size=32 * 4,
            modalities=["M"],
            nb_nodes_in_layer=256,
            nb_layers=4,
            optimizer_name="adam",
            learning_rate=0.00001,
            dropout_rate=0.0,
            label_smoothing=0.1,
            weight_decay=0.0,
            loss_name="binary_crossentropy",
            main_data_dir=r"/projects/palimpsests",
            palimpsest_name=r"Verona_msXL",
            folios_train=["msXL_335v_b", r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
                          "msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b", "msXL_334v_b", "msXL_344r_b",
                          "msXL_344v_b"],
            folios_val=[r"msXL_315r_b"],
            learning_rate_decay_epoch_step=0,
            patience=15,
            classes_dict={"undertext_renn": 1, "not_undertext": 0},            restore_path=None,
            debugging=False):
    """
    Trains a machine learning model on a specified dataset using given hyperparameters
    and saves the trained model to a specified directory.

    Parameters:
        current_time (str): Timestamp used for naming directories, defaults to current time.
        model_dir (str): Directory path where the trained model and logs will be saved.
        EPOCHS (int): Number of training epochs.
        batch_size (int): Batch size for training.
        modalities (list): List of data modalities to be used in the dataset.
        nb_nodes_in_layer (int): Number of nodes in each layer of the model.
        nb_layers (int): Number of layers in the model architecture.
        optimizer_name (str): Name of the optimizer to be used for training.
        learning_rate (float): Learning rate for the optimizer.
        dropout_rate (float): Dropout rate used in the model to prevent overfitting.
        label_smoothing (float): Smoothing parameter for labels during computation of loss.
        weight_decay (float): Weight decay parameter to regularize the model during training.
        loss_name (str): Loss function name to be used during training.
        main_data_dir (str): Path to the main directory containing the data.
        palimpsest_name (str): Name of the specific dataset being used for training.
        folios_train (list): List of training dataset folios.
        folios_val (list): List of validation dataset folios.
        learning_rate_decay_epoch_step (int): Step size for learning rate decay.
        patience (int): Number of epochs with no improvement to wait before early stopping.
        classes_dict (dict): Mapping of class names to corresponding numerical target values.
        restore_path (str or None): Path to restore a pre-trained model, if available.
        debugging (bool): Flag for enabling debugging mode during training.

    Raises:
        ValueError: If directories cannot be created or data cannot be loaded.

    Functionality:
        - Organizes the data into training and validation datasets.
        - Initializes and configures the graph-based model with specified hyperparameters.
        - Saves training and model-related parameters.
        - Configures callbacks for TensorBoard logging and early stopping.
        - Trains the model using the specified training data for the defined epochs.
        - Saves the trained model to the file system.
    """
    base_data_dir = osp(main_data_dir, palimpsest_name)
    model_dir = os.path.join(model_dir, palimpsest_name, current_time)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_dataset_par(folios_train,folios_val,model_dir,classes_dict)
    dataset_train, dataset_validation = dataset(base_data_dir,folios_train,folios_val,classes_dict,modalities)
    nb_features = dataset_train[0].shape[-1]

    print("nb_features",nb_features)
    gr = PalGraph(nb_features,nb_nodes_in_layer,model_dir,nb_layers,restore_path,optimizer_name,
                label_smoothing,loss_name,
                  dropout_rate,learning_rate)
    #save model hyperparametrs
    save_training_parameters(gr, debugging, batch_size, epochs,nb_features,
                           learning_rate_decay_epoch_step,dropout_rate,label_smoothing,weight_decay,patience)
    # same label distribution as in the train set
    log_dir = os.path.join(model_dir,'logs')
    #create the TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #create the Early Stopping callback
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
    history = gr.model.fit(dataset_train[0], dataset_train[1],
    epochs = epochs,
    callbacks = [tensorboard_callback,earlystopping_callback],
    validation_data = (dataset_validation[0], dataset_validation[1]),)

    gr.model.save(os.path.join(gr.model_dir, "model.keras"))



if __name__=="__main__":
  training(None,False)
