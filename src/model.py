import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.saving import register_keras_serializable
from keras.layers import Dense, Dropout, BatchNormalization, ReLU

def build_model(nb_features,nb_units_per_layer,nb_layers):
  inputs = keras.Input(shape=(nb_features,), name="counts")
  for layer_idx in range(nb_layers):
    if layer_idx==0:
      x = Dense(nb_units_per_layer, activation=None, name="dense_{}".format(layer_idx))(inputs)
    else:
      x = Dense(nb_units_per_layer, activation=None, name="dense_{}".format(layer_idx))(x)
    x = BatchNormalization(axis=1,name="batchnorm_{}".format(layer_idx))(x)
    x = ReLU(name="relu_{}".format(layer_idx))(x)
    #x = Dropout(0.5,name="dropout_{}".format(layer_idx))(x)
  outputs = Dense(1, name="predictions",activation=None)(x)
  return keras.Model(inputs=inputs, outputs=outputs)

@register_keras_serializable(package="MyModel", name="FCModel")
class FCModel(Model):
  def __init__(self,nb_features,nb_layers, **kwargs):
    super().__init__()
    self.nb_features = nb_features
    self.nb_layers = nb_layers
    self.layers_dict = {}
    for layer_idx in range(self.nb_layers):
      self.layers_dict["Dense_"+str(layer_idx)] = Dense(nb_features)
      self.layers_dict["Dropout_"+str(layer_idx)] = Dropout(0.5)
      self.layers_dict["BatchNorm_"+str(layer_idx)] = BatchNormalization(axis=1)
      self.layers_dict["Relu_" + str(layer_idx)] = ReLU()
    self.layers_dict["Last_layer"] = Dense(1)

  def call(self, x, training=None):
    for layer_idx in range(self.nb_layers):
      x = self.layers_dict["Dense_"+str(layer_idx)](x)
      x = self.layers_dict["BatchNorm_" + str(layer_idx)](x, training=training)
      x = self.layers_dict["Relu_" + str(layer_idx)](x)
      x = self.layers_dict["Dropout_"+str(layer_idx)](x,training=training)

    x = self.layers_dict["Last_layer"](x)
    return x

  def get_config(self):
    base_config = super().get_config()
    config = {"nb_features": self.nb_features,"nb_layers":self.nb_layers}
    return {**base_config, **config}



class SimpleModel(Model):
  def __init__(self,nb_features):
    super().__init__()
    self.d ={}
    self.d["d1"] = Dense(nb_features, activation='relu')
    self.d["dropout"] = Dropout(0.5)
    self.d["fc"] = Dense(1)

  @tf.function
  def call(self,x,training=False):
    x = self.d["d1"](x)
    x = self.d["dropout"](x)
    x = self.d["fc"](x)
    return x


if __name__=="__main__":
  net = SimpleModel(3)
  a = tf.constant([[1,2,3]])
  a_true = tf.constant([[1,2,3]])

  loss_obj = tf.keras.losses.MeanSquaredError()
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    out = net(a)
    loss = loss_obj(a_true,out)
  gradients = tape.gradient(loss, net.trainable_variables)
  print(net.trainable_variables)
  for var, g in zip(net.trainable_variables, gradients):
    print(f'{var.name}, grad val: {g.numpy()}')