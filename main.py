from NeuralNetworkTrainer import NeuralNetworkTrainer
from UtilityFunctions import UtilityFunctions
from DataProcessor import DataProcessor
import sys
import argparse
import numpy as np
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def main():

    parser = argparse.ArgumentParser(description=("arguments"))
    parser.add_argument("--epochs", default="10")
    parser.add_argument("--patience_epochs", default="10")
    parser.add_argument("--train_fraction", default="0.8")
    parser.add_argument("--seed", default="12345")
    parser.add_argument("--redshift", default="4")
    parser.add_argument("--fwhm", default="6")
    parser.add_argument("--hubble", default="0.676")
    parser.add_argument("--omegam", default="0.305147")
    parser.add_argument("--skewer_length", default="20")
    parser.add_argument("--bins", default="1024")
    parser.add_argument("--mean_flux", default=None)
    parser.add_argument("--noise", default="0.02")

    parser.add_argument("--seed_int", default="12345")
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--output_dir", default="output")
    
    parser.add_argument("--network", default="ConvNet")
    parser.add_argument("--batch_size", default="256")
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument('--features_per_block', action='store', 
                        default=[32, 64], 
                        type=int, nargs='*')
    
    parser.add_argument('--layers_per_block', action='store', 
                        default=[2, 2], 
                        type=int, nargs='*')


    args = parser.parse_args()

    patience_epochs = np.int32(args.patience_epochs)
    skewer_length = np.float32(args.skewer_length)
    train_fraction = np.float32(args.train_fraction)
    noise = np.float32(args.noise)

    hubble = np.float32(args.hubble)
    omegam = np.float32(args.omegam)

    bins = np.int32(args.bins)
    redshift = np.float32(args.redshift)
    fwhm = np.float32(args.fwhm)
    redshift = np.float32(args.redshift)

    seed_int = np.int32(args.seed_int)
    epochs = np.int32(args.epochs)
    epochs = np.int32(args.epochs)
    
    network = args.network
    batch_size = np.int32(args.batch_size)
    lr = np.float32(args.lr)
    layers_per_block = np.int32(args.layers_per_block)
    features_per_block = np.int32(args.features_per_block)

    utilities = UtilityFunctions()

    if args.mean_flux == None:
        mean_flux = utilities.mean_flux_z(redshift)
    else:
        mean_flux = np.float32(args.mean_flux)


    dataset_dir = args.dataset_dir+"/"
    output_dir = args.output_dir+"/"

    print('epochs, patience_epochs,  dataset_dir = ', epochs, 
          patience_epochs, dataset_dir
        )

    # collect, normalize and shape data for training and validation
    dp = DataProcessor(
        dataset_dir, output_dir, redshift, skewer_length, hubble, 
        omegam, fwhm, bins, mean_flux, seed_int
        )

    dp.make_dataset()

    nnt = NeuralNetworkTrainer(
         dp.get_output_dir(), "tempw", redshift, redshift, mean_flux, 
         fwhm, bins, seed_int, True, False)
    
    ds = dp.get_dataset()
    nnt.set_dataset(ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], 
                     noise, None, train_fraction
         )
    
    nnt.set_ml_model(network, layers_per_block, features_per_block)
    nnt.train(epochs, patience_epochs, batch_size, lr)
    nnt.predict()

###############################################################################


if __name__ == "__main__":
    # Check the number of arguments passed
    if len(sys.argv) > 20:
        print("Too many arguments..")
    else:
        main()
