import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, Conv2D, MaxPool2D
from noisy_labels.channel import Channel


def fcn_base(nb_layers,nb_units_per_layer,dropout_rate,batch_norm, inputs):
    """
    :param nb_layers: Number of fully connected layers in the network.
    :param nb_units_per_layer: Number of units (neurons) for each layer.
    :param dropout_rate: Dropout rate applied after each layer to reduce overfitting.
    :param batch_norm: Boolean indicating whether batch normalization should be applied after each dense layer.
    :param inputs: Input tensor to the fully connected layers.
    :return: Output tensor after applying all layers, batch normalization, activations, and dropout.
    """
    for layer_idx in range(nb_layers):
      if layer_idx==0:
        x = Dense(nb_units_per_layer, activation=None,kernel_initializer='glorot_uniform', name="dense_{}".format(layer_idx))(inputs)
      else:
        x = Dense(nb_units_per_layer, activation=None, kernel_initializer="he_normal", name="dense_{}".format(layer_idx))(x)
      if batch_norm:
        x = BatchNormalization(axis=1,name="batchnorm_{}".format(layer_idx))(x)
      x = ReLU(name="relu_{}".format(layer_idx))(x)
      x = Dropout(dropout_rate,name="dropout_{}".format(layer_idx))(x)
    return x


def ccn_base(nb_layers,nb_units_per_layer,dropout_rate,batch_norm, inputs):
    """
    :param nb_layers: Number of fully connected layers in the network.
    :param nb_units_per_layer: Number of units (neurons) for each layer.
    :param dropout_rate: Dropout rate applied after each layer to reduce overfitting.
    :param batch_norm: Boolean indicating whether batch normalization should be applied after each dense layer.
    :param inputs: Input tensor to the fully connected layers.
    :return: Output tensor after applying all layers, batch normalization, activations, and dropout.
    """
    
    for layer_idx in range(nb_layers):
      if layer_idx==0:
        x = Conv2D(nb_units_per_layer, kernel_size=3,padding="same", activation=None,kernel_initializer='glorot_uniform', name="dense_{}".format(layer_idx))(inputs)
      else:
        x = Conv2D(nb_units_per_layer, kernel_size=3,padding="same", activation=None, kernel_initializer="he_normal", name="dense_{}".format(layer_idx))(x)
      x = MaxPool2D(pool_size=(2, 2))(x)
      if batch_norm:
        x = BatchNormalization(axis=1,name="batchnorm_{}".format(layer_idx))(x)
      x = ReLU(name="relu_{}".format(layer_idx))(x)
      x = Dropout(dropout_rate,name="dropout_{}".format(layer_idx))(x)
    x = tf.keras.layers.Flatten()(x)
    return x

def build_model(nb_features,nb_units_per_layer,nb_layers,dropout_rate,batch_norm=False):
    """
    :param nb_features: Number of features in the input data.
    :param nb_units_per_layer: Number of units (neurons) in each dense layer.
    :param nb_layers: Number of dense layers to be added to the model.
    :param dropout_rate: Rate of dropout applied after each layer, as a float between 0 and 1.
    :param batch_norm: Boolean flag indicating whether to apply batch normalization after each dense layer.
    :return: A compiled Keras `Model` instance based on the provided layer specifications.
    """
    inputs = Input(shape=(nb_features,), name="input")
    x = fcn_base(nb_layers,nb_units_per_layer,dropout_rate,batch_norm, inputs)
    outputs = Dense(1, name="predictions", activation="sigmoid",kernel_initializer="he_normal",)(x)
    return Model(inputs=inputs, outputs=outputs)

def build_model_multiclass(nb_features,nb_units_per_layer,nb_layers,dropout_rate,nb_classes,win,batch_norm=False):
    """
    :param nb_classes: number of classes in the classification task.
    :param nb_features: Number of features in the input data.
    :param nb_units_per_layer: Number of units (neurons) in each dense layer.
    :param nb_layers: Number of dense layers to be added to the model.
    :param dropout_rate: Rate of dropout applied after each layer, as a float between 0 and 1.
    :param batch_norm: Boolean flag indicating whether to apply batch normalization after each dense layer.
    :return: A compiled Keras `Model` instance based on the provided layer specifications.
    """

    if win>0:
        inputs = Input(shape=(win+1,win+1,nb_features,), name="input")
        x = ccn_base(nb_layers,nb_units_per_layer,dropout_rate,batch_norm, inputs)
    elif win<=0:
        inputs = Input(shape=(nb_features,), name="counts")
        x = fcn_base(nb_layers,nb_units_per_layer,dropout_rate,batch_norm, inputs)
    else:
        raise ValueError("win should be a float number or int number, but got {}".format(win.__class__))

    outputs = Dense(nb_classes, name="predictions",activation="softmax",kernel_initializer="he_normal",)(x)
    return Model(inputs=inputs, outputs=outputs)

def build_model_with_noise_channel(model,channel_weights):
    """
    :param model: The original model whose output will be extended with a noise channel.
    :param confusion_matrix: A confusion matrix representing probabilities of misclassification, used to create the noise channel.
    :return: A new model incorporating the noisy channel and the original model's output.
    """
    baseline_output = model.layers[-1].output
    print(baseline_output)
    channeled_output = Channel(name='channel', weights=[channel_weights], units=baseline_output.shape[-1])(baseline_output)
    print("Channel output shape",channeled_output.shape)
    return Model(inputs=model.input, outputs=[channeled_output, baseline_output])