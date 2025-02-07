import argparse
from training_binary_new import training

def run_training():



    parser = argparse.ArgumentParser(
                        prog='Run training',
                        description='Trains supervised NN algorithm for palimpsest ink detection',
                        epilog='Thanks to the universe')
    parser.add_argument('--epochs',"-ep", type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size',"-bs", type=int, default=32 * 4, help='Batch size for training')
    parser.add_argument('--modalities',"-md", nargs='+', default=["M"], help='List of modalities')
    parser.add_argument('--nb_nodes_in_layer',"-nodes", type=int, default=256, help='Number of nodes in a layer')
    parser.add_argument('--nb_layers', "-layers", type=int, default=4, help='Number of layers')
    parser.add_argument('--optimizer_name', "-opt", type=str, default="adam", help='Name of the optimizer')
    parser.add_argument('--learning_rate', "-lr", type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--dropout_rate', "-dropout", type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing value')
    parser.add_argument('--weight_decay', "-wdecay", type=float, default=0.0, help='Weight decay value')
    parser.add_argument('--loss_name', "-loss",type=str, default="binary_crossentropy", help='Loss function name')
    parser.add_argument('--main_data_dir',"-datadir", type=str, default=r"/projects/palimpsests", help='Main data directory path')
    parser.add_argument('--palimpsest_name', "-pname", type=str, default=r"Verona_msXL", help='Palimpsest name')
    parser.add_argument('--folios_train',"-ftrain", nargs='+', default=["msXL_335v_b", r"msXL_315v_b", "msXL_318r_b",
                                                              "msXL_318v_b", "msXL_319r_b", "msXL_319v_b",
                                                              "msXL_322r_b", "msXL_322v_b", "msXL_323r_b",
                                                              "msXL_334r_b", "msXL_334v_b", "msXL_344r_b",
                                                              "msXL_344v_b"],
                        help='List of training folios')
    parser.add_argument('--folios_val',"-fval", nargs='+', default=[r"msXL_315r_b"], help='List of validation folios')

    parser.add_argument('--model_dir',"-mdir", type=str, default=None, help='Palimpsest model parent directory')
    parser.add_argument('--learning_rate_decay_epoch_step', type=int, default=0, help='Learning rate decay step')
    parser.add_argument('--classes_dict', type=dict, default={"undertext": 1, "not_undertext": 0},
                        help='Classes dictionary')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience parameter')

    args = parser.parse_args()



    # Pass all parsed arguments to the training function as keyword arguments
    training(
        epochs=args.EPOCHS,
        batch_size=args.batch_size,
        modalities=args.modalities,
        nb_nodes_in_layer=args.nb_nodes_in_layer,
        nb_layers=args.nb_layers,
        optimizer_name=args.optimizer_name,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        loss_name=args.loss_name,
        main_data_dir=args.main_data_dir,
        palimpsest_name=args.palimpsest_name,
        folios_train=args.folios_train,
        folios_val=args.folios_val,
        model_dir=args.model_dir,
        learning_rate_decay_epoch_step=args.learning_rate_decay_epoch_step,
        classes_dict=args.classes_dict,
        patience=args.patience
    )
