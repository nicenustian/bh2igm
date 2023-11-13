from neural_network_trainer import NeuralNetworkTrainer
from utility_functions import UtilityFunctions
from data_processor import DataProcessor
import sys
import argparse
import numpy as np
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def main():

    parser = argparse.ArgumentParser(description=("arguments"))
    parser.add_argument("--epochs", default="10")
    parser.add_argument("--patience_epochs", default="20")
    parser.add_argument("--seed", default="12345")
    parser.add_argument("--redshift", default="4")
    parser.add_argument("--fwhm", default="6")
    parser.add_argument("--hubble", default="0.676")
    parser.add_argument("--omegam", default="0.305147")
    parser.add_argument("--skewer_length", default="20")
    parser.add_argument("--bins", default="1024")
    parser.add_argument("--mean_flux", default=None)

    parser.add_argument("--seed_int", default="12345")
    parser.add_argument("--dataset_dir", default="skewers")
    parser.add_argument("--output_dir", default="output")

    args = parser.parse_args()

    patience_epochs = np.int32(args.patience_epochs)
    skewer_length = np.float32(args.skewer_length)
    hubble = np.float32(args.hubble)
    omegam = np.float32(args.omegam)

    bins = np.int32(args.bins)
    redshift = np.float32(args.redshift)
    fwhm = np.float32(args.fwhm)
    redshift = np.float32(args.redshift)

    seed_int = np.int32(args.seed_int)
    epochs = np.int32(args.epochs)

    utilities = UtilityFunctions()

    if args.mean_flux == None:
        mean_flux = utilities.mean_flux_z(redshift)
    else:
        mean_flux = np.float32(args.mean_flux)

    dataset_dir = args.dataset_dir+"/"
    output_dir = args.output_dir+"/"


    print('epochs, patience_epochs,  dataset_dir = ',
          epochs, patience_epochs, dataset_dir)

    # collect, normalize and shape data for traning and validation
    data_processor = DataProcessor(dataset_dir, output_dir, redshift,
                                   skewer_length, hubble, omegam, fwhm, bins,
                                   mean_flux, seed_int)

    flux, densityw, tempw, weights, flux_scaler_mean, flux_scaler_var = \
        data_processor.make_dataset()

    neural_network_trainer = NeuralNetworkTrainer(
        output_dir, "densityw", redshift, redshift, fwhm, bins,
        flux, densityw, tempw, weights,
        flux_scaler_mean, flux_scaler_var, noise_model=0.01)

    neural_network_trainer.train()

###############################################################################


if __name__ == "__main__":
    # Check the number of arguments passed
    if len(sys.argv) > 20:
        print("Too many arguments..")
    else:
        main()
