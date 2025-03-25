import argparse
from training_binary_channel import training
import json
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_training():
    parser = argparse.ArgumentParser(
                        prog='Run training',
                        description='Trains supervised NN algorithm for palimpsest ink detection',
                        epilog='Thanks to the universe')
    parser.add_argument('--epochs',"-ep", type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size',"-bs", type=int, default=32 * 4, help='Batch size for training')
    parser.add_argument('--modalities',"-md", nargs='+', default=["M"], help='List of modalities')
    parser.add_argument('--nb_nodes_in_layer',"-nodes", type=int, default=256, help='Number of nodes in a layer')
    parser.add_argument('--nb_layers', "-layers", type=int, default=4, help='Number of layers')
    parser.add_argument('--optimizer_name', "-opt", type=str, default="adam", help='Name of the optimizer')
    parser.add_argument('--learning_rate', "-lr", type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--dropout_rate', "-dropout", type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing value')
    parser.add_argument('--weight_decay', "-wdecay", type=float, default=0.0, help='Weight decay value')
    parser.add_argument('--loss_name', "-loss",type=str, default="sparce_categorical_crossentropy", help='Loss function name')
    parser.add_argument('--main_data_dir',"-datadir", type=str, default=r"D:", help='Main data directory path')
    parser.add_argument('--palimpsest_name', "-pname", type=str, default=r"Verona_msXL", help='Palimpsest name')
    parser.add_argument('--folios_train',"-ftrain", nargs='+', default=["msXL_335v_b"],
                        help='List of training folios')
    parser.add_argument('--folios_val',"-fval", nargs='+', default=[r"msXL_315r_b"], help='List of validation folios')

    parser.add_argument('--model_dir',"-mdir", type=str, default=r"c:\Data\PhD\ML_palimpsests\Supervised_palimpsest\training", help='Palimpsest model parent directory')
    parser.add_argument('--learning_rate_decay_epoch_step', type=int, default=0, help='Learning rate decay step')
    parser.add_argument('--classes_dict', type=str, default='{"undertext": 1, "not_undertext": 0}',
                        help='Classes dictionary')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience parameter')
    parser.add_argument('--window', type=int, default=10, help='Size of window around the pixel of interest, the whole patch areas is ((win+1)*(win+1))')
    parser.add_argument("--add_noise_channels", "-nch", action='store_true', default=False, help='Enable or disable noise channel')
    parser.add_argument("--restore_path", "-rep", type=str, default=None,
                        help='Enable or disable noise channel')
    parser.add_argument("--debug", "-deb", action='store_true', default=True, help='Reduce dataset size for debugging')

    args = parser.parse_args()
    # Validate and normalize paths
    args.model_dir = os.path.normpath(args.model_dir)
    # Convert the JSON string to a Python dictionary
    classes_dict = json.loads(args.classes_dict)
    args.classes_dict = classes_dict

    # Pass all parsed arguments to the training function as keyword arguments
    for key, value in args.__dict__.items():
        print(key, value, type(value))
    training(**args.__dict__)

if __name__ == "__main__":
    run_training()