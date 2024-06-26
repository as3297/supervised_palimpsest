import numpy as np
import tensorflow as tf
from model import FCModel,build_model
print("TensorFlow version:", tf.__version__)
from util import extend_json
from dataset import add_classes_in_split, shuffle_between_epoch, load_data_for_training
from datetime import datetime
import os

osp = os.path.join
#tf.compat.v1.disable_eager_execution()


class PalGraph():
  def __init__(self,nb_features,nb_units_per_layer,model_dir,nb_layers,restore_path,optimizer_name,label_smoothing,loss):
    # Create an instance of the model
    self.nb_units_per_layer = nb_units_per_layer
    self.nb_layers = nb_layers
    self.restore_path = restore_path
    if loss == "binary_crossentropy":
      self.loss_object = tf.keras.losses.BinaryCrossentropy(
          from_logits=True,
          label_smoothing=label_smoothing,
          axis=-1,
          reduction='sum_over_batch_size',
          name='binary_crossentropy'
      )
    elif loss == "hinge":
      self.loss_object = tf.keras.losses.Hinge(
    reduction='sum_over_batch_size', name='hinge'
)
    if optimizer_name == "adam":
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    elif optimizer_name == "sgd":
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.SGD(learning_rate= self.learning_rate)
    if restore_path is None:
      self.model = build_model(nb_features,nb_units_per_layer, nb_layers)
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




def save_training_parameters(gr,debugging,batch_size,nb_epochs,nb_features,learning_rate_decay_epoch_step):
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
  extend_json(osp(gr.model_dir,"training_parameters.json"),d)


def early_stoppping(val_loss_list,val_loss_cur,epoch,epoch_threshold):
    loss_goes_up=[]
    if epoch> 20:
        for old_loss in val_loss_list[-epoch_threshold:]:
            if val_loss_cur<old_loss:
                loss_goes_up.append(False)
        return np.all(loss_goes_up)
    else:
        return False


def training(restore_path = None,debugging=False):
  EPOCHS = 100
  batch_size = 32*8
  modalities = ["M"]
  nb_nodes_in_layer = 30
  nb_layers = 1
  optimizer_name = "adam"
  label_smoothing = 0.4
  loss_name = "binary_crossentropy"
  current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
  model_dir = os.path.join(r"C:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training", current_time)
  learning_rate_decay_epoch_step = 0

  if not os.path.exists(model_dir):
      os.makedirs(model_dir)
  trainset_ut, trainset_nonut, valset_ut, valset_nonut = load_data_for_training(
      model_dir, modalities, debugging)
  nb_features = trainset_ut[0].shape[1]
  gr = PalGraph(nb_features,nb_nodes_in_layer,model_dir,nb_layers,restore_path,optimizer_name,label_smoothing,loss_name)

  save_training_parameters(gr, debugging, batch_size, EPOCHS,nb_features,
                           learning_rate_decay_epoch_step)
  val_losses=[]

  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    gr.train_loss.reset_state()
    gr.train_acc.reset_state()
    gr.test_loss.reset_state()
    gr.test_acc.reset_state()


    trainset,nb_train_samples = shuffle_between_epoch(trainset_ut,trainset_nonut)
    valset = add_classes_in_split(valset_ut,valset_nonut)
    nb_val_samples = len(valset[1])
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
    gr.model.save(os.path.join(gr.model_dir,"model.keras"))
    val_losses.append(gr.val_loss.result())
    #if early_stoppping(val_losses,gr.val_loss.result(),epoch,15):
        #break
    #np.save(osp(gr.model_dir, r'optimizer.npy'), gr.optimizer.get_weights())


def testing(saved_model_path):
  imported = tf.saved_model.load(saved_model_path)
  pass

if __name__=="__main__":
  training(None,False)
