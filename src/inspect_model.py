import tensorflow as tf
import numpy as np

saved_model_path = r"C:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training\20240524-215116\model.keras"
imported = tf.keras.models.load_model(saved_model_path)
#imported = tf.saved_model.load(saved_model_path)
#imported = imported.signatures["serving_default"]
#imported= imported(nb_features=10,nb_layers=5)
imported.summary()
layer_config = imported.get_config()
print(layer_config)
print(imported.trainable_variables)
for var in imported.trainable_variables:

  print(var.path, "\n")