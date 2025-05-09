import tensorflow as tf
from model import build_model_multiclass,build_model_with_noise_channel
from src.dataset_patches_tf import dataset_tf
from util import extend_json, save_json, convert_float_in_dict, load_channel_weights,calculate_confusion_matrix
from datetime import datetime
from src.dataset import dataset
import os
from noisy_labels.channel import Channel
from tensorflow.keras.models import load_model
import numpy as np
osp = os.path.join


class PalGraph():
  def __init__(self,nb_features,nb_units_per_layer,model_dir,nb_layers,restore_path,
               optimizer_name,label_smoothing,loss,dropout_rate,learning_rate,add_noise_channels,nb_classes,win):
    # Create an instance of the model
    self.nb_units_per_layer = nb_units_per_layer
    self.nb_layers = nb_layers
    self.restore_path = restore_path
    self.learning_rate = learning_rate
    self.add_noise_channels = add_noise_channels
    self.model_dir = model_dir
    self.restore_path = restore_path
    self.win = win
    if loss == "binary_crossentropy":
      self.loss_object = tf.keras.losses.BinaryCrossentropy(
          from_logits=False,
          label_smoothing=label_smoothing,
          reduction='sum_over_batch_size',
          name='binary_crossentropy'
      )
    elif loss == "sparse_categorical_crossentropy":
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='sum_over_batch_size', name='sparse_categorical_crossentropy')
    else:
        raise ValueError(f"Loss function name {loss} not recognized")
    if optimizer_name == "adam":
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    elif optimizer_name == "sgd":
        self.optimizer = tf.keras.optimizers.SGD(learning_rate= self.learning_rate)

    if restore_path is None:
          self.model = build_model_multiclass(nb_features,nb_units_per_layer, nb_layers, dropout_rate,nb_classes,self.win)
          self.model.compile(
              optimizer=self.optimizer,
              loss=self.loss_object,
              loss_weights=None,
              metrics=["accuracy"],
          )
          self.model.summary()
    else:
        if add_noise_channels:

            pretrained_model = load_model(os.path.join(restore_path,"model.keras"),custom_objects={"Channel": Channel})
            channel_weights = load_channel_weights(restore_path)
            self.model = build_model_with_noise_channel(pretrained_model,channel_weights=channel_weights)
            # ignore baseline loss in training
            BETA = 0.
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_object,
                loss_weights=[1. - BETA, BETA],
                metrics=["accuracy"],
            )
        else:
            self.model = load_model(os.path.join(restore_path, 'model.keras'))
        print("Restored model from the path: ", self.restore_path, " ...")


    def restore_optimizer_state(self,model,optimizer):
        """If model stored in .h5 restore optimizer state first"""
        checkpoint = tf.train.Checkpoint(model=self.model,optim=self.optimizer)
        checkpoint.restore(path=os.path.join(self.restore_path,'ckpt-1')).assert_consumed()


def save_training_parameters(gr,batch_size,nb_epochs,nb_features,learning_rate_decay_epoch_step,
                             dropout_rate,label_smoothing,weight_decay,patience,win):
  d = {}
  d["restore_path"] = gr.restore_path
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
  d["add_noise_channels"] = gr.add_noise_channels
  d["patch_size"] = int(win+1)
  if win > 0:
      d["model_base_class"] = "cnn"
  else:
      d["model_base_class"] = "mlp"

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

def calculate_nb_samples_every_class(labels):
    # Get unique values and their counts
    unique_values, counts = np.unique(labels, return_counts=True)

    # Print counts
    for value, count in zip(unique_values, counts):
        print(f"Value {value} appears {count} times")

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
            label_smoothing=0.0,
            window = 10,
            weight_decay=0.0,
            loss_name="sparce_categorical_crossentropy",
            main_data_dir=r"/projects/palimpsests",
            palimpsest_name=r"Verona_msXL",
            folios_train=["msXL_335v_b",],# r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
                          #"msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b", "msXL_334v_b", "msXL_344r_b",
                          #"msXL_344v_b"],
            folios_val=[r"msXL_315r_b"],
            learning_rate_decay_epoch_step=0,
            patience=15,
            add_noise_channels = False,
            classes_dict={"undertext_renn": 1, "not_undertext": 0},
            restore_path=None):
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
    if window <=0 :
        dataset_train, dataset_validation = dataset(base_data_dir,folios_train,folios_val,classes_dict,modalities,window)
    else:
        dataset_train = dataset_tf(base_data_dir,folios_train,classes_dict,modalities,window,batch_size,shuffle=True,buffer_size=1000,box=None)
        dataset_validation = dataset_tf(base_data_dir,folios_val,classes_dict,modalities,window,batch_size,shuffle=False,buffer_size=1000,box=None)

    if window <= 0:
        if len(folios_val)==0:
            # Shuffle the training dataset with consistent order for features and labels
            shuffled_indices = np.random.permutation(len(dataset_train[0]))
            indices_val = shuffled_indices[:int(0.2*len(dataset_train[0]))]
            indices_train = shuffled_indices[int(0.2*len(dataset_train[0])):]
            dataset_validation = (dataset_train[0][indices_val], dataset_train[1][indices_val])
            dataset_train = (dataset_train[0][indices_train], dataset_train[1][indices_train])
        else:
            # Shuffle the training dataset with consistent order for features and labels
            shuffled_indices = np.random.permutation(len(dataset_train[0]))
            dataset_train = (dataset_train[0][shuffled_indices], dataset_train[1][shuffled_indices])

    sample = next(iter(dataset_train))  # e.g., (feature, label)
    features, labels = sample
    print("Feature shape",features.shape)
    print("Label shape",labels.shape)
    nb_features = features.shape[-1]
    gr = PalGraph(nb_features,nb_nodes_in_layer,model_dir,nb_layers,restore_path,optimizer_name,
                label_smoothing,loss_name,
                  dropout_rate,learning_rate,add_noise_channels,len(classes_dict.keys()),window)
    #save model hyperparametrs
    save_training_parameters(gr, batch_size, epochs,nb_features,
                           learning_rate_decay_epoch_step,dropout_rate,label_smoothing,weight_decay,patience,window)
    # same label distribution as in the train set
    log_dir = os.path.join(model_dir,'logs')
    #create the TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,update_freq='epoch')
    #create the Early Stopping callback
    if patience==-1:
        patience=epochs
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
    if window>0:
        history = gr.model.fit(dataset_train,
                               batch_size=batch_size,
                               epochs=epochs,
                               callbacks=[tensorboard_callback, earlystopping_callback],
                               validation_data=dataset_validation)
    else:
        history = gr.model.fit(dataset_train[0], dataset_train[1],
        batch_size = batch_size,
        epochs = epochs,
        callbacks = [tensorboard_callback,earlystopping_callback],
        validation_data = (dataset_validation[0], dataset_validation[1]),shuffle=True,)

    gr.model.save(os.path.join(gr.model_dir, "model.keras"))
    #save model as h5
    gr.model.save(os.path.join(gr.model_dir, "model.h5"))


    #save test and train metrics
    train_metrics = {
        "loss": history.history.get('loss'),
        "accuracy": history.history.get('accuracy')
    }
    val_metrics = {
        "val_loss": history.history.get('val_loss'),
        "val_accuracy": history.history.get('val_accuracy')
    }
    a = "noise_channel" if add_noise_channels else ""
    metrics_path = os.path.join(gr.model_dir, "train_val_loss_history" + a + ".json")
    save_json( metrics_path,{"train_metrics": train_metrics, "val_metrics": val_metrics},)
    if not add_noise_channels:
        confusion_matrix = calculate_confusion_matrix(
            gr.model,gr.model_dir,
            dataset_train[0],
            dataset_train[1],
            batch_size,len(classes_dict.keys()))
        print("Confusion matrix",confusion_matrix)
    #save model architecture to json
    json_config = gr.model.to_json()
    save_path = os.path.join(gr.model_dir, "model_architecture.json")
    save_json(save_path, json_config)


if __name__=="__main__":
  training(None,False)
