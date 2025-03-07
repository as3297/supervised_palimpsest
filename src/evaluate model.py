import json

import argparse
import os
import tensorflow as tf
from util import read_json

def load_model(restore_path):
    model = tf.keras.models.load_model(os.path.join(restore_path, 'model.keras'))
    d = read_json(os.path.join(restore_path,"training_parameters.json"))
    loss = d["loss"]
    label_smoothing = None
    if loss == "binary_crossentropy":
      loss_object = tf.keras.losses.BinaryCrossentropy(
          from_logits=False,
          label_smoothing=label_smoothing,
          reduction='sum_over_batch_size',
          name='binary_crossentropy'
      )

    optimizer_config = json.loads(d["optimizer"])
    optimizer_name = optimizer_config["name"]
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam.from_config(optimizer_config)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD.from_config(optimizer_config)
    model.compile(
        optimizer=optimizer,
        loss=loss_object,
        loss_weights=None,
        metrics=["accuracy"],
    )
    return model
def eval_train_val():
    pass
def run_training():
    parser = argparse.ArgumentParser(
                        prog='Run training',
                        description='Trains supervised NN algorithm for palimpsest ink detection',
                        epilog='Thanks to the universe')
    parser.add_argument('--model_dir',"-mdir", type=str, default=r"c:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training", help='Palimpsest model parent directory')

    args = parser.parse_args()
    # Validate and normalize paths
    args.model_dir = os.path.normpath(args.model_dir)


    model = load_model(args.model_dir)

if __name__ == "__main__":
    run_training()