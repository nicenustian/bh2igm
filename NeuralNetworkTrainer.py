import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import keras.backend as K
from machine_learning_models.ConvNet import ConvNet
from machine_learning_models.ResNet import ResNet
from numpy import random
from UtilityFunctions import UtilityFunctions
from QuantityPlotter import QuantityPlotter

class NeuralNetworkTrainer:
    def __init__(self, output_dir, quantity, redshift, obs_redshifts, mean_flux,
                 fwhm, bins, seed, not_observational=True, load_best_model=False
                 ):
        
        self.output_dir = output_dir
        self.quantity = quantity
        self.redshift = redshift
        self.obs_redshifts = obs_redshifts
        self.fwhm = fwhm
        self.bins = bins
        self.mean_flux = mean_flux
        self.seed = seed
        self.not_observational = not_observational
        self.load_best_model = load_best_model
        self.set_seed()
        self.uf = UtilityFunctions()
        self.qp = QuantityPlotter(2, 1, 28, 6, 20, 0.67)

        
    def set_seed(self):
      random.seed(self.seed)
      np.random.seed(self.seed)
      tf.random.set_seed(self.seed)
      tf.experimental.numpy.random.seed(self.seed)
      tf.random.set_seed(self.seed)
      
      os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
      os.environ['TF_DETERMINISTIC_OPS'] = '1'
      os.environ["PYTHONHASHSEED"] = str(self.seed)
      print(f"Random seed set as {self.seed}")
      
      
    def set_noise(self):        
        
        utilities = UtilityFunctions()
            
        if isinstance(self.noise_model, np.float32):
            self.snr = 1/self.noise_model
            print('fixed snr=', self.snr)
            self.noise = np.full(self.flux.shape, self.noise_model, dtype=np.float64)
        else:
            self.snr = np.int32(np.mean(1.0/self.noise_model[self.noise_model>0]))
            self.noise = np.zeros(self.flux.shape, dtype=np.float64)
            
            #normalised the flux bins
            self.flux_bins = (self.flux_bins - self.flux_mean)/self.flux_var
            for i in range(self.Ntotal):
                #get the corresponding noise for each flux pixel
                self.noise[i] = self.noise_model[utilities.closest_argmin(self.flux[i], self.flux_bins)]
                #adjust the flux pixels to one, si they become featureless regions
                self.flux[i][self.bad==1] = 1.0
                #noise to zero in bad pixels regions
                self.noise[i][self.bad==1] = 0.0
  
    
    def set_ml_model(self, network, layers_per_block, features_per_block): 
        
        self.network = network
        self.layers_per_block = layers_per_block
        self.features_per_block = features_per_block
            
        if 'ResNet' == self.network:
            
            self.ml_model  = ResNet(self.layers_per_block, 
                                    self.features_per_block, 
                                    self. Nnodes, self.seed)

        elif  'ConvNet' == self.network:
            self.ml_model  = ConvNet(self.layers_per_block, 
                                     self.features_per_block, 
                                     self.Nnodes, self.seed)

        else:
            raise ValueError('Unknown Network: {}'.format(self.network))
                
        print("network, layers, features, units, lr, batch_size = ", 
                  self.network, self.layers_per_block, self.features_per_block, 
                  self. Nnodes)


    def set_dataset(self, flux, densityw, tempw, weights, scalar_mean, scalar_var, 
                    noise_model, flux_bins, train_fraction
                  ):
      
      self.flux = flux
      self.densityw = densityw
      self.tempw = tempw
      self.weights = weights
  
      self.noise_model = noise_model
      self.flux_bins = flux_bins      
      self.scalar_mean = scalar_mean
      self.scalar_var = scalar_var
      self.train_fraction = train_fraction
      
      self.Npixels = self.flux.shape[1]
      self.Ntotal = self.flux.shape[0]
      self.Nnodes = self.Npixels
      self.Ntrain = np.int32(self.Ntotal*self.train_fraction)
      self.Ntest = self.Ntotal - self.Ntrain
      
      self.set_noise()
      self.post_output = '_snr'+str(np.int32(self.snr))+'_mflux'+"{:.4f}".format(self.mean_flux)+\
      '_fwhm'+"{:.2f}".format(self.fwhm)+'_z'+"{:.2f}".format(self.obs_redshifts)

      
      if self.quantity=='densityw':
          yy = self.densityw
      elif self.quantity=='tempw':
          yy = self.tempw
      else:
          raise ValueError('Unknown quantity: {}'.format(self.quantity))       
   
      self.flux = np.expand_dims(self.flux, axis=2)
      self.noise = np.expand_dims(self.noise, axis=2)
  
      self.train_data = tf.data.Dataset.from_tensor_slices((
          self.flux[:self.Ntrain], yy[:self.Ntrain], 
          self.noise[:self.Ntrain],  self.weights[:self.Ntrain]))
      self.test_data = tf.data.Dataset.from_tensor_slices((
          self.flux[self.Ntrain:], yy[self.Ntrain:], 
          self.noise[self.Ntrain:],  self.weights[self.Ntrain:]))

        
    @tf.function
    def rolling(self, x_input, shifts):
            return tf.vectorized_map(
                lambda x: tf.roll(x[0], shift=x[1], axis=0),
                elems=[x_input, shifts])

    @tf.function
    def mae_func(self, y_true, y_pred):
            return K.mean(K.abs(y_true - y_pred))


    @tf.function
    def nll_func(self, y_true, y_pred, weights):
            nll = -y_pred.log_prob(y_true)
            nll *= weights
            nll = K.sum(nll)
            return nll


    @tf.function
    def sigma_cover(self, y_true, y_pred, sigma_pred):
            y_pred_upper = tf.reshape(y_pred + sigma_pred, [-1])
            y_pred_lower = tf.reshape(y_pred - sigma_pred, [-1])
            y_true = tf.reshape(y_true, [-1])
            return tf.math.count_nonzero((y_true>=y_pred_lower) & 
                                      (y_true<=y_pred_upper))

    
    @tf.function
    def train_model(self, x, y, w):
        with tf.GradientTape() as tape:
            y_pred = self.ml_model(x, training=True)
            loss_nll = self.nll_func(tf.cast(y, dtype=self.type_casting), y_pred, 
                           tf.cast(w, dtype=self.type_casting))/self.Ntrain
            
            loss_kll = tf.reduce_sum(self.ml_model.losses)/self.kll_fact
            loss = loss_nll + loss_kll
 
        grads = tape.gradient(loss, self.ml_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.ml_model.trainable_weights))
        
        count = self.sigma_cover(
            tf.cast(y, dtype=self.type_casting), y_pred.mean(),
            y_pred.stddev())
        
        self.mae.update_state(self.mae_func(
            tf.cast(y, dtype=self.type_casting), y_pred.mean())
            )
        self.nll_sum.update_state(loss_nll)
        self.kll_sum.update_state(loss_kll)
        self.count_sum.update_state(count)
        

    @tf.function
    def test_model(self, x_test, y_test, w_test):
        y_pred_test = self.ml_model(x_test, training=False)
        loss_nll_test = self.nll_func(tf.cast(y_test, dtype=self.type_casting), y_pred_test,
                       tf.cast(w_test, dtype=self.type_casting))/self.Ntest
        loss_kll_test = tf.reduce_sum(self.ml_model.losses)
        
        count_test = self.sigma_cover(tf.cast(y_test, dtype=self.type_casting), 
                                 y_pred_test.mean(), y_pred_test.stddev())
        self.test_mae.update_state(
            self.mae_func(tf.cast(y_test, dtype=self.type_casting), y_pred_test.mean())
            )
        self.test_nll_sum.update_state(loss_nll_test)
        self.test_kll_sum.update_state(loss_kll_test)
        self.test_count_sum.update_state(count_test)

    
    @tf.function
    def train_on_batches(self):
        
        for step, (x_batch, y_batch, noise_batch, 
                   w_batch) in enumerate(self.train_data):
            
            # Assuming x_batch has shape (batch_size, ...
            batch_size = tf.shape(x_batch)[0] 
            
            shifts = tf.random.uniform(
                shape=(batch_size,), maxval=self.Npixels, 
                dtype=tf.int32, seed=self.seed)
            
            x_batch = self.rolling(x_batch, shifts)
            y_batch = self.rolling(y_batch, shifts)
                        
            x_batch += ((noise_batch/np.sqrt(self.scalar_var[0]))*tf.random.normal(
                (batch_size, self.Npixels, 1), 0, 1, tf.float64, seed=self.seed))
 
            self.train_model(x_batch, y_batch, w_batch)
           
            
    @tf.function
    def test_on_batches(self):
        for step, (x_batch_test, y_batch_test, noise_batch_test, 
                   w_batch_test) in enumerate(self.test_data):
            
            # Assuming x_batch has shape (batch_size, ...
            batch_size = tf.shape(x_batch_test)[0]  
            
            shifts = tf.random.uniform(
                shape=(batch_size,), maxval=self.Npixels, dtype=tf.int32, 
                seed=self.seed)
            
            x_batch_test = self.rolling(x_batch_test, shifts)
            y_batch_test = self.rolling(y_batch_test, shifts)
            
            x_batch_test += ((noise_batch_test/np.sqrt(self.scalar_var[0]))*tf.random.normal(
                (batch_size, self.Npixels, 1), 0, 1, tf.float64, seed=self.seed))
 
            self.test_model(x_batch_test, y_batch_test, w_batch_test)
    
    def initialize_metrics(self):
        
        self.best_metric = np.Infinity
        self.current_metric = np.Infinity
        
        self.mae = tf.keras.metrics.Mean()
        self.nll_sum = tf.keras.metrics.Sum()
        self.kll_sum = tf.keras.metrics.Sum()
        self.count_sum = tf.keras.metrics.Sum()
        
        self.test_mae = tf.keras.metrics.Mean()
        self.test_nll_sum = tf.keras.metrics.Sum()
        self.test_kll_sum = tf.keras.metrics.Sum()
        self.test_count_sum = tf.keras.metrics.Sum()
        
        self.loss_nll_list = []
        self.loss_kll_list = []
        self.mae_list = []
        self.count_list = []
        
        self.test_loss_nll_list = []
        self.test_loss_kll_list = []
        self.test_mae_list = []
        self.test_count_list = []

        self.type_casting = tf.float32
        
            
    @tf.function
    def reset_metrics(self):                           
        self.nll_sum.reset_state()
        self.kll_sum.reset_state()
        self.mae.reset_state()
        self.count_sum.reset_state()
        
        self.test_nll_sum.reset_state()
        self.test_kll_sum.reset_state()
        self.test_mae.reset_state()
        self.test_count_sum.reset_state()
        
        
    def update_metrics(self):
        self.loss_nll_list.append(self.nll_sum.result().numpy()
                                  /self.Npixels)
        self.loss_kll_list.append(self.kll_sum.result().numpy()
                                  /self.Npixels)
        self.mae_list.append(self.mae.result().numpy())
        self.count_list.append(self.count_sum.result().numpy()
                               /self.Ntrain/self.Nnodes)
            
        self.test_loss_nll_list.append(self.test_nll_sum.result().numpy()
                                      /self.Npixels)
        self.test_loss_kll_list.append(self.test_kll_sum.result().numpy()
                                      /self.Npixels)
        self.test_mae_list.append(self.test_mae.result().numpy())
        self.test_count_list.append(self.test_count_sum.result().numpy()
                                   /self.Ntest/self.Nnodes)
        
        self.current_metric = self.test_nll_sum.result().numpy() + \
            self.test_kll_sum.result().numpy()
                
        if self.current_metric <= self.best_metric:
                self.no_improvement_count = 0
                self.best_metric = self.current_metric

                print()
                weights_filename = self.output_dir+'weights_'+self.quantity+'/'
                print('saving weights.. improved from', 
                      self.best_metric, 'to', self.current_metric, 
                      weights_filename)
                self.ml_model.save_weights(weights_filename)
        else:
                self.no_improvement_count += 1
        
    
    def print_metrics(self, time_in_sec):
        print()
        print('Epoch', self.epoch, np.int32(time_in_sec),'[sec]', 
                      ' improve_count =', self.no_improvement_count)
       
        if len(self.loss_nll_list)>0:
            print('train', "nll = {:f}".format(self.loss_nll_list[-1]), 
                          "mae = {:f}".format(self.mae_list[-1]), 
                          "sigma_cov = {:f}".format(self.count_list[-1]))
            
            if self.loss_kll_list[-1] !=0:
                      print("kll = {:f}".format(self.loss_kll_list[-1]))

        
        if len(self.test_loss_nll_list)>0:
            print(' test',"nll = {:f}".format(self.test_loss_nll_list[-1]), 
                          "mae = {:f}".format(self.test_mae_list[-1]), 
                          "sigma_cov = {:f}".format(self.test_count_list[-1]))
            if self.test_loss_kll_list[-1] !=0:
                      print("kll = {:f}".format(self.test_loss_kll_list[-1]))
        
        
    def save_history_file(self):
        #convert to numpy arrays and save the loss values
        history_filename = self.output_dir+'history_'+self.quantity+'.npy'
        print('saving ', history_filename)
        with open(history_filename, 'wb') as f:
            #the train metrics
            np.save(f, np.array(self.loss_nll_list, dtype=np.float32))
            np.save(f, np.array(self.loss_kll_list, dtype=np.float32))
            np.save(f, np.array(self.mae_list, dtype=np.float32))
            np.save(f, np.array(self.count_list, dtype=np.float32))
            #the test metrics
            np.save(f, np.array(self.test_loss_nll_list, dtype=np.float32))
            np.save(f, np.array(self.test_loss_kll_list, dtype=np.float32))
            np.save(f, np.array(self.test_mae_list, dtype=np.float32))
            np.save(f, np.array(self.test_count_list, dtype=np.float32))

    def predict(self):
        
        file_list = self.uf.get_files_list_from_dir(self.output_dir, "processed_", ".npy")
        self.load_best_weights()
        
        for model in file_list:
        
            filename = self.output_dir+model
            print('opening processed skewers file', filename)
  
            try:
                with open(filename, 'rb') as f:
                    flux = np.load(f)
                    densityw = np.load(f)
                    tempw = np.load(f)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                        
                
            flux = (flux - self.scalar_mean[0]) / np.sqrt(self.scalar_var[0])
            flux = np.expand_dims(flux, axis=2)
            
            dist = self.ml_model(tf.convert_to_tensor(flux), training=False)
            mean = dist.mean().numpy()
            std = dist.stddev().numpy()
    
            upper = mean + std
            lower = mean - std
            
            if self.quantity=='densityw':
                index = 1
            else:
                index = 2
                
    
            mean = (mean * np.sqrt(self.scalar_var[index])) + self.scalar_mean[index]
            upper = (upper * np.sqrt(self.scalar_var[index])) + self.scalar_mean[index]
            lower = (lower * np.sqrt(self.scalar_var[index])) + self.scalar_mean[index]
            
            flux = flux * np.sqrt(self.scalar_var[0]) + self.scalar_mean[0]
    
            print('quantity actual and predicted')
            if self.not_observational:
                if self.quantity=='densityw':
                    print('pred=',10**np.quantile(mean, [0.14, 0.5, 0.84]))
                    print('actu=',10**np.quantile(densityw, [0.14, 0.5, 0.84]))
                else:
                    print('pred=',10**np.quantile(mean, [0.14, 0.5, 0.84]))
                    print('actu=',10**np.quantile(tempw, [0.14, 0.5, 0.84]))
            
        
            self.qp.set_dataset(flux, tempw, mean, upper, lower)
            self.qp.plot_los(0)
            self.qp.save_plot(self.output_dir+'los_'+self.quantity+'_'+model)
            self.qp.clear_curves()
                    
            filename = self.output_dir+'predicted_'+self.quantity+'_'+model
            print('saving ', filename)
            with open(filename, 'wb') as f:
                  np.save(f, mean)
                  np.save(f, upper)
                  np.save(f, lower)

    def load_best_weights(self):
        if self.load_best_model==True:
            ml_model_filename = self.output_dir+'nnweights_'+self.quantity+self.post_output+'/'
            print('loading model ', ml_model_filename)
            try:
                self.ml_model.load_weights(ml_model_filename).expect_partial()
            except FileNotFoundError as e:
                print(f"Error: {e}")


    def train(self, epochs, patience_epochs, batch_size, lr):
        
        self.load_best_weights()
        self.epoch = 0
        self.no_improvement_count = 0
        self.patience_epochs = patience_epochs
        self.batch_size = batch_size
        self.lr  = lr
        self.epochs = epochs
        self.kll_fact = 1
                  
        self.train_data = self.train_data.shuffle(self.Ntrain).batch(self.batch_size)
        self.test_data = self.test_data.shuffle(self.Ntest).batch(self.batch_size)
        
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.ml_model.compile(optimizer=self.optimizer)
                
        self.initialize_metrics()
        
        while ((self.epoch < self.epochs) and
        (self.no_improvement_count < self.patience_epochs)):
            start =  time.time()
            
            self.epoch += 1
            self.train_on_batches()
            self.test_on_batches()
            self.update_metrics()
            self.print_metrics(time.time()-start)     
            self.reset_metrics()
            
        self.save_history_file()
            