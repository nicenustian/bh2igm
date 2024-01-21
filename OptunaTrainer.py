import joblib
import optuna
import numpy as np
from NeuralNetworkTrainer import NeuralNetworkTrainer


class OptunaTrainer:
    def __init__(self, output_dir, files_list, filename, load_study, 
                 input_quantity, output_quantity,
                 seed_int, trails, epochs, patience_epochs,
                 train_fraction, dataset, post_file_name, noise):
        
        self.output_dir = output_dir
        self.files_list = files_list
        self.filename = filename
        self.load_study = load_study

        
        self.input_quantity = input_quantity
        self.output_quantity = output_quantity
        
        self.trails = trails
        self.max_layers_per_block = 4
        self.epochs = epochs
        self.patience_epochs =  patience_epochs

        self.dataset = dataset
        self.post_file_name = post_file_name
        self.noise = noise
        self.train_fraction  = train_fraction
        self.seed = seed_int
        self.load_study_file()


    def load_study_file(self):
        
        self.study_file = self.output_dir+"/"+self.filename+"_"+self.input_quantity+'_'+\
            self.output_quantity+".pkl"
        
        if self.load_study==True:
            print('loading pickle file', self.study_file)
            self.study = joblib.load(self.study_file)
                
            print("Best trial until now:")
            print(" Value: ", self.study.best_trial.value)
            print(" Params: ")
            for key, value in self.study.best_trial.params.items():
                print(f"{key}: {value}")
    
        else:
            print('created study file')
            self.study = optuna.create_study(direction="minimize")
            
        
        optuna.logging.set_verbosity(optuna.logging.FATAL)
        
            
    def suggest_hyperparams(self):
                
        #suggest a network
        self.network = self.trial.suggest_categorical("network", ["MLPNet"])#,"ConvNet", "ResNet"])
        
        #choose hyper parameters for model training
        self.lr = self.trial.suggest_float('lr', 1e-4, 0.5, log=True)
        # batch size and features per block in power od two
        self.batch_size = 2**self.trial.suggest_int("batch_size", 8, 10)
        self.num_blocks = self.trial.suggest_int("num_blocks", 1, 6)
        
        self.layers_per_block = np.ones(self.num_blocks, dtype=np.int32)
        features_log2 = np.ones(self.num_blocks, dtype=np.int32)
        
        self.layers_per_block[0] = self.trial.suggest_int("layers_per_block1", 1, 2)
        features_log2[0] = self.trial.suggest_int("features_per_block1", 1, 5)
        
    
        for ci in range(1, self.num_blocks):

            self.layers_per_block[ci] = self.trial.suggest_int("layers_per_block"+str(ci+1), 
                self.layers_per_block[ci-1], self.max_layers_per_block)
            
            features_log2[ci] = self.trial.suggest_int("features_per_block"+str(ci+1), 
                                   features_log2[ci-1], 
                                   features_log2[ci-1]+1)
            
        
        self.features_per_block = np.int32(2**features_log2)
        

    def print_best_callback(self, study, trial):
        print()
        print('trail number =', len(self.study.trials))
        print(f"Best value: {self.study.best_value}, Best params: {self.study.best_trial.params}")
        print('saving file..', self.study_file)
        joblib.dump(self.study, self.study_file)


    def wrapper(self, trial):
        
        print()
        print()
        
        self.trial  = trial
        self.suggest_hyperparams()

        self.nnt = NeuralNetworkTrainer(self.output_dir, self.network, self.seed, False,
                                        self.input_quantity, self.output_quantity)
        self.nnt.set_dataset(self.dataset, self.files_list, self.post_file_name, 
                             self.noise, 
                      None, None, self.train_fraction)
        
        self.nnt.set_ml_model(self.layers_per_block, 
                              self.features_per_block)
        best_metric = self.nnt.train(False, self.epochs, self.patience_epochs, 
                                     self.batch_size, self.lr)
        
                
        return best_metric
    
    
    def run_trails(self):
        
        self.study.optimize(self.wrapper, n_trials=self.trails, 
                            callbacks=[self.print_best_callback])
        optuna_dict = dict(self.study.best_trial.params.items())

        self.network = optuna_dict['network']
        self.lr = optuna_dict['lr']
        self.batch_size = 2**optuna_dict['batch_size']
        self.layers_per_block = optuna_dict['network']

        
        self.layers_per_block = np.ones(optuna_dict['num_blocks'], dtype=np.int32)
        self.features_per_block = np.ones(optuna_dict['num_blocks'], dtype=np.int32)
        
        self.layers_per_block[0] = optuna_dict['layers_per_block1']
        self.features_per_block[0] = 2**optuna_dict['features_per_block1']
     
        
        for li in range(optuna_dict['num_blocks']):

            self.layers_per_block[li] = optuna_dict['layers_per_block'+str(li+1)]
            self.features_per_block[li] = 2**optuna_dict['features_per_block'+str(li+1)]
            
        print(self.network, self.lr, self.batch_size, \
              self.layers_per_block, self.features_per_block)
        
        return self.network, self.lr, self.batch_size, \
              self.layers_per_block, self.features_per_block
            
            
