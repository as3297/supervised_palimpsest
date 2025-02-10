#!/bin/bash

# Set default values for the arguments.
EPOCHS=500
BATCH_SIZE=512
MODALITIES=("M")
NB_NODES_IN_LAYER=512
NB_LAYERS=4
OPTIMIZER_NAME="adam"
LEARNING_RATE=0.00001
DROPOUT_RATE=0.0
LABEL_SMOOTHING=0.1
WEIGHT_DECAY=0.0
LOSS_NAME="binary_crossentropy"
MAIN_DATA_DIR="/projects/palimpsests" #"D:"
PALIMPSEST_NAME="Verona_msXL"
FOLIOS_TRAIN=("msXL_335v_b" "msXL_315v_b" "msXL_318r_b" "msXL_318v_b" "msXL_319r_b" "msXL_319v_b" "msXL_322r_b" "msXL_322v_b" "msXL_323r_b" "msXL_334r_b" "msXL_334v_b" "msXL_344r_b")
FOLIOS_VAL=("msXL_344v_b") #("msXL_315r_b")
MODEL_DIR="/projects/supervised_palimpsest/training" #"c:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training"
LEARNING_RATE_DECAY_EPOCH_STEP=0
CLASSES_DICT='{"undertext_renn":1,"not_undertext":0}'
PATIENCE=15

python run_training.py --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --modalities "${MODALITIES[@]}" \
  --nb_nodes_in_layer $NB_NODES_IN_LAYER \
  --nb_layers $NB_LAYERS \
  --optimizer_name $OPTIMIZER_NAME \
  --learning_rate $LEARNING_RATE \
  --dropout_rate $DROPOUT_RATE \
  --label_smoothing $LABEL_SMOOTHING \
  --weight_decay $WEIGHT_DECAY \
  --loss_name $LOSS_NAME \
  --main_data_dir $MAIN_DATA_DIR \
  --palimpsest_name $PALIMPSEST_NAME \
  --folios_train "${FOLIOS_TRAIN[@]}" \
  --folios_val "${FOLIOS_VAL[@]}" \
  --model_dir $MODEL_DIR \
  --learning_rate_decay_epoch_step $LEARNING_RATE_DECAY_EPOCH_STEP \
  --classes_dict $CLASSES_DICT \
  --patience $PATIENCE > logs.txt
