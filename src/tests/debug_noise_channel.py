import os, sys
# Add the root directory (one level up) to the module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from util import calculate_confusion_matrix, load_channel_weights
import numpy as np
from dataset import dataset
import os

from tensorflow.keras.models import load_model



restore_path = r'c:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training\Verona_msXL\20250307-144511'
noise_weights = load_channel_weights(restore_path)
main_data_dir=r"d:"
palimpsest_name=r"Verona_msXL"
data_base_dir = r"D:\Verona_msXL"
folios_train=["msXL_335v_b",]# r"msXL_315v_b", "msXL_318r_b", "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
                          #"msXL_322r_b", "msXL_322v_b", "msXL_323r_b", "msXL_334r_b", "msXL_334v_b", "msXL_344r_b",
                          #"msXL_344v_b"],
folios_val=[r"msXL_315r_b"]
classes_dict={"undertext_renn": 1, "not_undertext": 0}
modalities=["M"]
# Load the Keras model from the specified folder
model_path = os.path.join(restore_path, "model.h5")
model = load_model(model_path)
dataset_train, dataset_validation = dataset(data_base_dir ,folios_train,folios_val,classes_dict,modalities,True)
Y_train = np.squeeze(dataset_train[1])
X_train = dataset_train[0]

predictions = model.predict(dataset_train[0])
predictions = np.argmax(predictions, axis=-1)
# To understand the model's predictions, you can compare them with true labels
baseline_confusion = np.zeros((2, 2))
print("pred_labels",predictions)
print("_______________")
print("org_labels",Y_train)
for n, p in zip(Y_train,predictions):
    baseline_confusion[p, n] += 1.
print(baseline_confusion)


