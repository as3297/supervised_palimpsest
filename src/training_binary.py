from cProfile import label

import numpy as np
import tensorflow as tf
from model import FCModel,build_model
print("TensorFlow version:", tf.__version__)
from util import extend_json, save_json
from dataset import add_classes_in_split, shuffle_between_epoch, load_data_for_training, resample_nb_dataset_points
from datetime import datetime
import os
#from tf.keras.losses import get_regularization_loss
osp = os.path.join
#tf.compat.v1.disable_eager_execution()


class PalGraph():
  def __init__(self,nb_features,nb_units_per_layer,model_dir,nb_layers,restore_path,
               optimizer_name,label_smoothing,loss,class_balansing_factor,
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
    elif loss == "binary_focal_crossentropy":
      self.loss_object = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=class_balansing_factor,
            gamma=2.0,
            from_logits=False,
            label_smoothing=label_smoothing,
            axis=-1,
            reduction="sum_over_batch_size",
            name="binary_focal_crossentropy",
            dtype=None,
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

    self.variable_names = [i.path for i in self.model.trainable_variables]
    # metric variables
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.val_loss = tf.keras.metrics.Mean(name='val_loss')
    self.test_loss = tf.keras.metrics.Mean(name='test_loss')
    self.train_acc = tf.keras.metrics.BinaryAccuracy(name='train_accuracy', dtype=None, threshold=0.5)
    self.val_acc = tf.keras.metrics.BinaryAccuracy(name='val_accuracy', dtype=None, threshold=0.5)
    self.test_acc = tf.keras.metrics.BinaryAccuracy(name='test_accuracy', dtype=None, threshold=0.5)
    self.train_prec = tf.keras.metrics.Precision(name='train_precision')
    self.val_prec = tf.keras.metrics.Precision(name='val_precision')
    self.test_prec = tf.keras.metrics.Precision(name='test_precision')

    # summary callbacks

    train_log_dir = self.model_dir+'/logs/gradient_tape/train'
    val_log_dir = self.model_dir+'/logs/gradient_tape/val'
    self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    #saving model

  @tf.function(input_signature=(tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None], dtype=tf.int32)))
  def train_step(self,images, labels):
    gradients,loss,predictions = self.get_gradients(images,labels)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    self.train_loss(loss)
    self.train_acc(labels, predictions)
    self.train_prec(labels,predictions)


  @tf.function(input_signature=(tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None], dtype=tf.int32)))
  def val_step(self,images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = self.model(images, training=False)
    t_loss = self.loss_object(labels, predictions)

    self.val_loss(t_loss)
    self.val_acc(labels, predictions)
    self.val_prec(labels, predictions)


  def get_gradients(self,images, labels):
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = self.model(images, training=True)
      loss = self.loss_object(labels, predictions,)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    return gradients,loss,predictions

def tensorboard_train(gr,gradients,variable_names,epoch):
  """
  Record training metrics for tensorboard during eager execution
  :param gr: object that holds the summary writer object
  :return:
  """
  with gr.train_summary_writer.as_default():
    tf.summary.scalar('loss', gr.train_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', gr.train_acc.result(), step=epoch)
    tf.summary.scalar("precision",gr.train_prec.result(),step=epoch)
    tf.summary.scalar("learning_rate", gr.optimizer.learning_rate.numpy(), step=epoch)
    if epoch%5==0:
        for grads in zip(variable_names,gradients):
          tf.summary.histogram(grads[0], grads[1], step=epoch)


def tensorboard_val(gr, epoch):
  """
  Record validation metrics for tensorboard during eager execution
  :param gr: object that holds the summary writer object
  :return:
  """
  with gr.val_summary_writer.as_default():
    tf.summary.scalar('loss', gr.val_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', gr.val_acc.result(), step=epoch)
    tf.summary.scalar("precision", gr.val_prec.result(), step=epoch)




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


class EarlyStopping:
    def __init__(self, patience,model):
        self.patience = patience
        self.wait = 0
        self.model = model
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.save_weights = False
        self.best_weights = None
        self.early_stop = False

    def early_stoppping(self,val_loss=None,val_acc=None):
        if val_loss is None and val_acc is None:
            raise ValueError("Either validation loss or validation accuracy should be provided for EarlyStoping")
        elif val_loss is not None and val_acc is not None:
            raise ValueError("Only one of validation loss or validation accuracy should be provided for EarlyStoping")
        criteria = False
        if val_acc is not None:
            criteria = val_acc > self.best_val_acc
            if criteria:
                self.best_val_acc = val_acc
        if val_loss is not None:
            criteria = val_loss < self.best_val_loss
            if criteria:
                self.best_val_loss = val_loss
        if criteria:
            self.wait = 0  # reset wait counter
            self.best_weights = self.model.get_weights()  # save best model weightstraining_binary.py
        else:
            self.wait += 1
            print(f'Validation loss did not improve for {self.wait} epochs.')
            # If validation loss hasn't improved for 'patience' epochs, stop training
        if self.wait >= self.patience:
            print('Early stopping.')
            self.model.set_weights(self.best_weights)  # restore best model weights
            self.early_stop = True


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
    main_data_dir = r"/projects/palimpsests"
    palimpsest_dir = r"Verona_msXL"
    base_data_dir = osp(main_data_dir,palimpsest_dir)
    folios = ["msXL_344v_b","msXL_344r_b","msXL_335v_b","msXL_335r_b","msXL_334v_b","msXL_334r_b","msXL_323v_b",
              "msXL_323r_b","msXL_322v_b","msXL_322r_b","msXL_319v_b","msXL_319r_b","msXL_318v_b","msXL_318r_b",
              "msXL_315v_b","msXL_315r_b"]
    ut_mask_file = r"undertext_cleaned_10nn_ot_sub_black"
    nonut_mask_file = r"bg_lines_ot_subtracted_black"
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(r"/projects/palimp/training",palimpsest_dir, current_time)
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
    nb_features = trainset_ut[0].shape[1]
    gr = PalGraph(nb_features,nb_nodes_in_layer,model_dir,nb_layers,restore_path,optimizer_name,
                label_smoothing,loss_name,class_balansing_factor,
                  weight_decay,dropout_rate,learning_rate)
    #initialize early stopping
    early_stop = EarlyStopping(patience,gr.model)
    #save model hyperparametrs
    save_training_parameters(gr, debugging, batch_size, EPOCHS,nb_features,
                           learning_rate_decay_epoch_step,dropout_rate)
    #equalize the number of points in the validation set to achieve
    # same label distribution as in the train set
    valset = resample_nb_dataset_points(valset_ut, valset_nonut)
    nb_val_samples = len(valset[1])
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        gr.train_loss.reset_state()
        gr.train_acc.reset_state()
        gr.test_loss.reset_state()
        gr.test_acc.reset_state()


        trainset,nb_train_samples = shuffle_between_epoch(trainset_ut,trainset_nonut)

        gradients_tensorboard = []

        for idx in range(0,nb_train_samples,batch_size):
            train_batch = tf.constant(trainset[0][idx:idx+batch_size,:], dtype=tf.float32)
            labels = tf.constant(trainset[1][idx:idx+batch_size],dtype=tf.int32)
            gr.train_step(train_batch,labels)
            if idx==0:
                gradients_tensorboard,_,_ = gr.get_gradients(train_batch,labels)
            elif idx%300==0:
                gradients, _, _ = gr.get_gradients(train_batch, labels)
                for i,var_grad in enumerate(gradients):
                    gradients_tensorboard[i] = tf.concat([gradients_tensorboard[i], var_grad],axis=0)
        tensorboard_train(gr,gradients_tensorboard,gr.variable_names,epoch)


        for idx in range(0,nb_val_samples,batch_size):
            val_batch = tf.constant(valset[0][idx:idx+batch_size,:],dtype=tf.float32)
            gr.val_step(val_batch, tf.constant(valset[1][idx:idx+batch_size],dtype=tf.int32))
        tensorboard_val(gr,epoch)

        if learning_rate_decay_epoch_step>0:
          if epoch%learning_rate_decay_epoch_step==0 and epoch>0:
            gr.learning_rate = gr.learning_rate / 10
            gr.optimizer.learning_rate.assign(gr.learning_rate)

        print(
          f'Epoch {epoch + 1}, '
          f'Loss: {gr.train_loss.result():0.3f}, '
          f'Accuracy: {gr.train_acc.result() * 100:0.2f}, '
          f'Val Loss: {gr.val_loss.result():0.3f}, '
          f'Val Accuracy: {gr.val_acc.result() * 100:0.2f}'
        )

        early_stop.early_stoppping( val_loss=None,val_acc=gr.val_acc.result())
        if early_stop.early_stop and epoch>50:
            break
        #np.save(osp(gr.model_dir, r'optimizer.npy'), gr.optimizer.get_weights())
        #save model
        gr.model.save(os.path.join(gr.model_dir, "model.keras"))

def testing(saved_model_path):
  imported = tf.saved_model.load(saved_model_path)
  pass

if __name__=="__main__":
  training(None,False)
