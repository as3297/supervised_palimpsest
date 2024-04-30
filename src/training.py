import tensorflow as tf
from model import FCModel
from read_pixel_coord import read_points_class
import numpy as np
print("TensorFlow version:", tf.__version__)
from read_msi_image import PointsfromMSI_PIL


test_bbox = [60, 1197, 4680, 2288]
train_bbox = [60, 2288, 4680, 5857]
val_bbox = [60, 5857, 4680, 6441]


def sublist_of_bands(bands,modality="M"):
  bands_subset = []
  for band in bands:
    if "-"+modality in band:
      bands_subset.append(band)
  return bands_subset


def data_from_file(im_path_ut,im_path_notut):

  train_feat_ut, train_labels_ut = read_points_class(train_bbox,im_path_ut)
  train_feat_nout, train_labels_nout = read_points_class(train_bbox,im_path_notut)

  val_feat_ut, val_labels_ut = read_points_class(val_bbox,im_path_ut)
  val_feat_nout, val_labels_nout = read_points_class(val_bbox,im_path_notut)

  test_feat_ut, test_labels_ut = read_points_class(test_bbox,im_path_ut)
  test_feat_nout, test_labels_nout  = read_points_class(test_bbox,im_path_notut)

  return None



# Create an instance of the model
model = FCModel(nb_features)

loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,
    label_smoothing=0.0,
    axis=-1,
    reduction='sum_over_batch_size',
    name='binary_crossentropy'
)
optimizer = tf.keras.optimizers.Adam()

# metric variables
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
train_acc = tf.keras.metrics.BinaryAccuracy(name='train_accuracy', dtype=None, threshold=0.5)
val_acc = tf.keras.metrics.BinaryAccuracy(name='val_accuracy', dtype=None, threshold=0.5)
test_acc = tf.keras.metrics.BinaryAccuracy(name='test_accuracy', dtype=None, threshold=0.5)
# summary callbacs

#saving model

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_acc(labels, predictions)

@tf.function
def val_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  val_loss(t_loss)
  val_acc(labels, predictions)


def training(train_ds,val_ds):

  EPOCHS = 5

  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_state()
    train_acc.reset_state()
    test_loss.reset_state()
    test_acc.reset_state()

    for images, labels in train_ds:
      train_step(images, labels)

    for test_images, test_labels in val_ds:
      val_step(test_images, test_labels)

    print(
      f'Epoch {epoch + 1}, '
      f'Loss: {train_loss.result():0.2f}, '
      f'Accuracy: {train_acc.result() * 100:0.2f}, '
      f'Test Loss: {val_loss.result():0.2f}, '
      f'Test Accuracy: {val_acc.result() * 100:0.2f}'
    )