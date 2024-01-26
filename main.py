from NeuralNetworkTrainer import NeuralNetworkTrainer
from UtilityFunctions import UtilityFunctions
from DataProcessor import DataProcessor
from OptunaTrainer import OptunaTrainer
import sys
import argparse
import numpy as np
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def main():

    # I/O SEARCH PARAMS
    parser = argparse.ArgumentParser(description=("arguments"))
    parser.add_argument("--input_quantity", default="flux")
    parser.add_argument("--output_quantity", default="tempw")
    parser.add_argument("--output_dir", default="ml_outputs")
    parser.add_argument("--redshift", default="4")
    parser.add_argument("--dataset_dir", default="dataset_files")
    parser.add_argument("--dataset_file_filter", default="train")
    parser.add_argument("--prediction_file_filter", default="model")
    parser.add_argument("--train_fraction", default="0.8")
    parser.add_argument("--seed_int", default="12345")
    
    # GRID SEARCH PARAMS
    parser.add_argument("--grid_search", default=False)
    parser.add_argument("--load_study", action='store_true', default=False)
    parser.add_argument("--study_file", default="hyperparams_search")
    parser.add_argument("--trails", default="20")
    parser.add_argument("--search_epochs", default="20")
    parser.add_argument("--search_patience_epochs", default="20")


    # TRAIN SEARCH PARAMS
    # default hyper params if the no grid search is enabled
    # otherwise these params are set by Optuna grid search
    parser.add_argument("--epochs", default="40")
    parser.add_argument("--patience_epochs", default="40")
    parser.add_argument('--load_best_model', action='store_true', default=False)
    
    
    # TRAIN SEARCH PARAMS -- NETWORK ARCHITECTURE
    # ResNET, ConvNet, MLPNet
    # If grid search is enabled these numbers are replaced by the 
    # results from Optuna search
    parser.add_argument("--network", default="ResNet")
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--batch_size", default="2048")
    parser.add_argument("--noweights", default=True)
    parser.add_argument('--layers_per_block', action='store',
                        default=[2,3,3, 4,4,4], type=int, nargs='*')
    parser.add_argument('--features_per_block', action='store',
                        default=[32,32,32, 32,32,64], type=int, nargs='*')
    
    # DATA PROCESSING PARAMS
    # standard processing params related to data processing
    parser.add_argument("--bins", default=None)
    parser.add_argument("--mean_flux", default=None)
    # Noise value to be added serves as one sigma level of
    # Gaussian noise with zero mean
    # Must be positive. 0 means no noise
    parser.add_argument("--noise", default="0.02")
    parser.add_argument("--fwhm", default="6")
    parser.add_argument("--quasar", default="J021043")
    parser.add_argument("--hubble", default="0.676")
    parser.add_argument("--omegam", default="0.305147")
    parser.add_argument("--skewer_length", default="20")


    args = parser.parse_args()
    input_quantity = args.input_quantity
    output_quantity = args.output_quantity
    patience_epochs = np.int32(args.patience_epochs)
    skewer_length = np.float32(args.skewer_length)
    train_fraction = np.float32(args.train_fraction)
    noise = np.float32(args.noise)
    

    hubble = np.float32(args.hubble)
    omegam = np.float32(args.omegam)

    redshift = np.float32(args.redshift)
    fwhm = np.float32(args.fwhm)
    redshift = np.float32(args.redshift)
    seed_int = np.int32(args.seed_int)
    epochs = np.int32(args.epochs)
    
    search_epochs = np.int32(args.search_epochs)
    search_patience_epochs = np.int32(args.search_patience_epochs)
    
    study_file = args.study_file
    trails = np.int32(args.trails)
    
    network = args.network
    batch_size = np.int32(args.batch_size)
    lr = np.float32(args.lr)
    layers_per_block = np.int32(args.layers_per_block)
    features_per_block = np.int32(args.features_per_block)

    utilities = UtilityFunctions()
        
    if args.bins is None:
        bins = utilities.fwhm_to_bins(fwhm, skewer_length, redshift, hubble, 
                                      omegam)
    else:
        bins = np.int32(args.bins)
        

    if args.mean_flux == None:
        mean_flux = utilities.mean_flux_z(redshift)
    else:
        mean_flux = np.float32(args.mean_flux)

    dataset_dir = args.dataset_dir+"/"
    output_dir = args.output_dir+"/"

    print('epochs, <F>, patience_epochs,  dataset_dir = ',
          epochs, mean_flux, patience_epochs, dataset_dir)

    ##################################################################################
    '''
    print()
    print('MAKING DATASET..')

    # collect, normalize and shape data for training and validation
    dp = DataProcessor(dataset_dir, args.dataset_file_filter, args.quasar,
                        output_dir, input_quantity, output_quantity, args.noweights,
                        redshift, skewer_length, hubble, omegam, fwhm, bins, mean_flux, 
                        noise, seed_int)

    dp.make_dataset(True)
    
    
    # if grid search is true replace the hyperparams using grid search
    if args.grid_search:
        print()
        print('GRID SEARCH..')
        opt = OptunaTrainer(dp.get_output_dir(), dp.get_files_list(), 
                            study_file, args.load_study, 
                            input_quantity, output_quantity,
                            seed_int, trails, 
                            search_epochs, search_patience_epochs, 
                            train_fraction, dp.get_dataset(), 
                            dp.get_post_file_name(), noise)
        
        network, lr, batch_size, layers_per_block, \
            features_per_block = opt.run_trails()
        del opt
    
    
    print()
    print('ML TRAINING..')
    nnt = NeuralNetworkTrainer(dp.get_output_dir(), redshift, network, 
                               seed_int, args.load_best_model,
                               input_quantity, output_quantity)
    nnt.set_dataset(dp.get_dataset(), dp.get_files_list(), 
                    dp.get_post_file_name(), noise, 
                      None, None, train_fraction)
        
    nnt.set_ml_model(layers_per_block, features_per_block)
    nnt.train(True, epochs, patience_epochs, batch_size, lr)
    
    del dp
    del nnt
    
    '''
    ###########################################################################
    
    
    print()
    print('PREDICTIONS..')
    
    # collect, normalize and shape data for predictions
    dp = DataProcessor(dataset_dir, args.prediction_file_filter, args.quasar,
                        output_dir, input_quantity, output_quantity, args.noweights,
                        redshift, skewer_length, hubble, omegam, fwhm, bins, mean_flux, 
                        noise, seed_int)
    # With no normalizing or shuffling
    dp.make_dataset(False)
    
    nnt = NeuralNetworkTrainer(dp.get_output_dir(), redshift, network, seed_int, True,
                                input_quantity, output_quantity)
    
    nnt.set_dataset(dp.get_dataset(), dp.get_files_list(),
                    dp.get_post_file_name(), noise, 
                      None, None, train_fraction)
        
    nnt.set_ml_model(layers_per_block, features_per_block)
    if args.quasar != None:
        nnt.predict_obs_los(dataset_dir, args.quasar)
    nnt.predict(dp)
    
    ############################################################################


if __name__ == "__main__":
    # Check the number of arguments passed
    if len(sys.argv) > 33:
        print("Too many arguments..")
    else:
        main()
