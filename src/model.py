import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU


def build_model(nb_features,nb_units_per_layer,nb_layers,dropout_rate,batch_norm=False):
  inputs = keras.Input(shape=(nb_features,), name="counts")
  for layer_idx in range(nb_layers):
    if layer_idx==0:
      x = Dense(nb_units_per_layer, activation=None,kernel_initializer='glorot_uniform', name="dense_{}".format(layer_idx))(inputs)
    else:
      x = Dense(nb_units_per_layer, activation=None, kernel_initializer="he_normal", name="dense_{}".format(layer_idx))(x)
    if batch_norm:
      x = BatchNormalization(axis=1,name="batchnorm_{}".format(layer_idx))(x)
    x = ReLU(name="relu_{}".format(layer_idx))(x)
    x = Dropout(dropout_rate,name="dropout_{}".format(layer_idx))(x)
  outputs = Dense(1, name="predictions",activation="sigmoid",kernel_initializer="he_normal",)(x)
  return keras.Model(inputs=inputs, outputs=outputs)

