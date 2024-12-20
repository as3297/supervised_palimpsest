from cProfile import label

import numpy as np
import tensorflow as tf
from model import FCModel,build_model
print("TensorFlow version:", tf.__version__)
from util import extend_json, save_json
from dataset import add_classes_in_split, shuffle_between_epoch, load_data_for_training, resample_nb_dataset_points
from datetime import datetime
import os
from dataset_tf import create_oversampled_dataset
#from tf.keras.losses import get_regularization_loss
osp = os.path.join
#tf.compat.v1.disable_eager_execution()


class PalGraph():
  def __init__(self,nb_features,nb_units_per_layer,model_dir,nb_layers,restore_path,
               optimizer_name,label_smoothing,loss,
               weight_decay,dropout_rate,learning_rate):
    # Create an instance of the model
    self.nb_units_per_layer = nb_units_per_layer
    self.nb_layers = nb_layers
    self.restore_path = restore_path
    self.learning_rate = learning_rate
    if loss == "binary_crossentropy":
      self.loss_object = tf.keras.losses.BinaryCrossentropy(
          from_logits=False,
          label_smoothing=label_smoothing,
          axis=-1,
          reduction='sum_over_batch_size',
          name='binary_crossentropy'
      )


    if optimizer_name == "adam":
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    elif optimizer_name == "sgd":
        self.optimizer = tf.keras.optimizers.SGD(learning_rate= self.learning_rate)
    elif optimizer_name == "adamw":
        #adam with weight decay
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.learning_rate,weight_decay=weight_decay)
    if restore_path is None:
      self.model = build_model(nb_features,nb_units_per_layer, nb_layers, dropout_rate)
      self.model_dir = model_dir
      self.model.compile(
    optimizer=self.optimizer,
    loss=self.loss_object,
    loss_weights=None,
    metrics=None,
    weighted_metrics=None,
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile='auto',
    auto_scale_loss=True
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
  save_path = osp(gr.model_dir,"training_parameters.json")
  if not os.path.exists(save_path):
     save_json(save_path, d)
  else:
    extend_json(save_path, d)



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
    nb_nodes_in_layer = 256
    nb_layers = 4
    optimizer_name = "adamw"
    weight_decay = 0.0
    learning_rate = 0.000001
    dropout_rate = 0.0
    label_smoothing = 0.1
    loss_name = "binary_crossentropy"
    main_data_dir = r"C:\Data\PhD\palimpsest\Victor_data"
    palimpsest_dir = r"Paris_Coislin"
    base_data_dir = osp(main_data_dir,palimpsest_dir)
    folios = [r"Par_coislin_393_054r"]
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(r"C:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training",palimpsest_dir, current_time)
    learning_rate_decay_epoch_step = 0
    #Early stopping parametrs
    patience = 15
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    dataset = create_oversampled_dataset(batch_size,folios,palimpsest_dir,modalities,shuffle=True)
    nb_features = dataset.element_spec[0].shape[-1]
    print("nb_features",nb_features)
    gr = PalGraph(nb_features,nb_nodes_in_layer,model_dir,nb_layers,restore_path,optimizer_name,
                label_smoothing,loss_name,
                  weight_decay,dropout_rate,learning_rate)
    #save model hyperparametrs
    save_training_parameters(gr, debugging, batch_size, EPOCHS,nb_features,
                           learning_rate_decay_epoch_step,dropout_rate)
    #equalize the number of points in the validation set to achieve
    # same label distribution as in the train set
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    history = gr.model.fit(dataset, validation_split=0.2, epochs=EPOCHS, batch_size=batch_size,callbacks=[callback])
    gr.model.save(os.path.join(gr.model_dir, "model.keras"))

def testing(saved_model_path):
  imported = tf.saved_model.load(saved_model_path)
  pass

if __name__=="__main__":
  training(None,False)
