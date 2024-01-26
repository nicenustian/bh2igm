import numpy as np
import os
import tensorflow as tf
import time
import matplotlib
import matplotlib.pyplot as plt
font = {'family' : 'serif', 'weight' : 'normal','size' : 34}
matplotlib.rc('font', **font)
import keras.backend as K
from machine_learning_models.ConvNet import ConvNet
from machine_learning_models.ResNet import ResNet
from machine_learning_models.MLPNet import MLPNet
from numpy import random
from UtilityFunctions import UtilityFunctions


class NeuralNetworkTrainer:
    def __init__(self, output_dir, redshift, network, seed, load_best_model,
                 input_quantity, output_quantity):
        
        self.output_dir = output_dir
        self.redshift = redshift
        self.seed = seed
        self.load_best_model = load_best_model
        self.input_quantity = input_quantity
        self.output_quantity = output_quantity
        self.network = network
        self.uf = UtilityFunctions()


        self.set_seed()
        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(
            self.strategy.num_replicas_in_sync)
            )
        
        
    def set_seed(self):
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        tf.experimental.numpy.random.seed(self.seed)

        # When running on the CuDNN backend, two further options must be set
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        print(f"Random seed set as {self.seed}")
         
    
    def set_ml_model(self, layers_per_block, features_per_block): 
        
        self.layers_per_block = layers_per_block
        self.features_per_block = features_per_block
        
            
        if 'ResNet' == self.network:
            with self.strategy.scope():
                self.ml_model  = ResNet(self.layers_per_block, 
                                        self.features_per_block, 
                                        self. Nnodes, self.seed)

        elif  'ConvNet' == self.network:
            with self.strategy.scope():
                self.ml_model  = ConvNet(self.layers_per_block, 
                                         self.features_per_block,
                                         self.Nnodes, self.seed)
                
        elif  'MLPNet' == self.network:
            with self.strategy.scope():
                self.ml_model  = MLPNet(self.layers_per_block,
                                        self.Nnodes, self.seed)

        else:
            raise ValueError('Unknown Network: {}'.format(self.network))
        
        if self.network == 'MLPNet':
            print("network, layers, units = ", 
                      self.network, np.sum(self.layers_per_block), 
                      self. Nnodes)
        else:    
            print("network, layers, features, units = ", 
                  self.network, self.layers_per_block, self.features_per_block, 
                  self. Nnodes)
        
        if self.load_best_model==True:
            ml_model_filename = self.output_dir+'nnweights_'+\
                self.input_quantity+"_"+self.output_quantity+'/'
            print('loading model ', ml_model_filename)
            self.ml_model.load_weights(ml_model_filename).expect_partial() 



    def set_dataset(self, ds, files_list, post_output, noise_model, 
                    flux_bins, bad, train_fraction):
      
      self.x = ds[0]
      self.y = ds[1]
      self.n = ds[2]
      self.w = ds[3]
      self.xscalar_mean = ds[4]
      self.xscalar_var = ds[5]
      
      self.files_list = files_list
      self.train_fraction = train_fraction
      
      self.Npixels = self.x.shape[1]
      self.Ntotal = self.x.shape[0]
      self.Nnodes = self.Npixels
      self.Ntrain = np.int32(self.Ntotal*self.train_fraction)
      self.Ntest = self.Ntotal - self.Ntrain
      
      self.post_output = post_output
   
      if self.network != "MLPNet":
          self.x = np.expand_dims(self.x, axis=2)    
          self.n = np.expand_dims(self.n, axis=2)
  
      self.train_data = tf.data.Dataset.from_tensor_slices((
          self.x[:self.Ntrain], self.y[:self.Ntrain], 
          self.n[:self.Ntrain],  self.w[:self.Ntrain]))
      
      self.test_data = tf.data.Dataset.from_tensor_slices((
          self.x[self.Ntrain:], self.y[self.Ntrain:], 
          self.n[self.Ntrain:],  self.w[self.Ntrain:]))
      
      
      options = tf.data.Options()
      options.experimental_distribute.auto_shard_policy = \
          tf.data.experimental.AutoShardPolicy.FILE
      self.train_data = self.train_data.with_options(options) 
      self.test_data = self.test_data.with_options(options)

        
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
            return tf.math.count_nonzero((y_true>=y_pred_lower) &\
                                         (y_true<=y_pred_upper))


    # Function to apply learning rate decay
    def apply_learning_rate_decay(self):
        self.lr *= 0.5
        self.no_improvement_count = 0
        self.optimizer.learning_rate.assign(self.lr)
    

    #@tf.function
    def train_model(self, x, y, n, w):
            
        shifts = tf.random.uniform(
                shape=(tf.shape(x)[0],), maxval=self.Npixels, 
                dtype=tf.int32, seed=self.seed)
            
        x = self.rolling(x, shifts)
        y = self.rolling(y, shifts)
        
        if self.input_quantity=="flux":
            x += ((n/np.sqrt(self.xscalar_var))*tf.random.normal(
                tf.shape(x), 0, 1, tf.float64, seed=self.seed))
        
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
    def distributed_train_model(self, x, y, n, w):
      self.strategy.run(self.train_model, args=(x, y, n, w))



    ##@tf.function
    def train_on_batches(self):

        for step, (x_batch, y_batch, noise_batch,
                    w_batch) in enumerate(self.train_data):

            self.distributed_train_model(x_batch, y_batch, noise_batch, w_batch)

        if self.no_improvement_count == 10:
            self.apply_learning_rate_decay()


    ## @tf.function
    def test_model(self, x, y, n, w):
            
        shifts = tf.random.uniform(
                shape=(tf.shape(x)[0],), maxval=self.Npixels, dtype=tf.int32, 
                seed=self.seed
                )
            
        x = self.rolling(x, shifts)
        y = self.rolling(y, shifts)
                        
        if self.input_quantity=="flux":
            x += ((n/np.sqrt(self.xscalar_var))*tf.random.normal(
                tf.shape(x), 0, 1, tf.float64, seed=self.seed))
        
        y_pred_test = self.ml_model(x, training=False)
        loss_nll_test = self.nll_func(tf.cast(y, dtype=self.type_casting), y_pred_test,
                       tf.cast(w, dtype=self.type_casting))/self.Ntest
            
        loss_kll_test = tf.reduce_sum(self.ml_model.losses)     
        
        count_test = self.sigma_cover(tf.cast(y, dtype=self.type_casting), 
                                 y_pred_test.mean(), y_pred_test.stddev())
        self.test_mae.update_state(
            self.mae_func(tf.cast(y, dtype=self.type_casting), y_pred_test.mean())
            )
        self.test_nll_sum.update_state(loss_nll_test)
        self.test_kll_sum.update_state(loss_kll_test)
        self.test_count_sum.update_state(count_test)

  
    
    @tf.function
    def distributed_test_model(self, x, y, n, w):
      self.strategy.run(self.test_model, args=(x, y, n, w))

                
    # @tf.function
    def test_on_batches(self):
        for step, (x,y,n,w) in enumerate(self.test_data):
            self.distributed_test_model(x,y,n,w)


    
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
        self.loss_nll_list.append(self.nll_sum.result().numpy()/self.Npixels)
        self.loss_kll_list.append(self.kll_sum.result().numpy()/self.Npixels)
        self.mae_list.append(self.mae.result().numpy())
        self.count_list.append(self.count_sum.result().numpy()/self.Ntrain/self.Nnodes)
            
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
                
                weights_filename = self.output_dir+'nnweights_'\
                    +self.input_quantity+"_"+self.output_quantity+'/'
                print()
                
                if self.save_weights==True:
                    print('saving weights.. improved from', 
                          self.best_metric, 'to', self.current_metric, 
                          weights_filename)
                    self.ml_model.save_weights(weights_filename)
                    
                self.best_metric = self.current_metric

        else:
                self.no_improvement_count += 1
        
    
    def print_metrics(self, time_in_sec):
        
        print()
        
        print('Epoch', self.epoch, np.int32(time_in_sec),'[sec]', ' lr=', self.lr,  
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
        history_filename = self.output_dir+'history_'+self.input_quantity+\
            "_"+self.output_quantity+'.npy'
            
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
    


    def train(self, save_weights, epochs, patience_epochs, 
              batch_size, lr):  

        self.save_weights = save_weights
        self.epoch = 0
        self.no_improvement_count = 0
        self.patience_epochs = patience_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.kll_fact = 1

            
        self.train_data = self.strategy.experimental_distribute_dataset(
             self.train_data
             #.map(self.train_data)
             .shuffle(self.Ntrain)
             .batch(self.batch_size, drop_remainder=True)
             .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
             )
        
        self.test_data = self.strategy.experimental_distribute_dataset(
            self.test_data
            #.map(self.test_data)
            .shuffle(self.Ntest)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
  
        
        with self.strategy.scope():
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
            self.ml_model.compile(optimizer=self.optimizer)
                
        self.initialize_metrics()
        
        print("lr, batch size, Ntrain, Ntest", 
              self.lr, self.batch_size, self.Ntrain, self.Ntest)
        print()

        
        while ((self.epoch < self.epochs) and
        (self.no_improvement_count < self.patience_epochs)):
            start =  time.time()
            
            self.epoch += 1
            self.train_on_batches()
            self.test_on_batches()
            end = time.time()
            self.update_metrics()
            self.print_metrics(end-start)
            self.reset_metrics()
        
        if self.save_weights==True:
            self.save_history_file()
        
        # Return best metric value
        return np.min(self.test_loss_nll_list)
    
    
    def read_scalars_file(self):
        
        save_file = self.output_dir+'scaler_'+self.input_quantity+'_'+self.output_quantity
        with open(save_file, 'rb') as f:
            self.xscalar_mean = np.load(f)
            self.xscalar_var = np.load(f)
            self.yscalar_mean = np.load(f)
            self.yscalar_var = np.load(f)
   
    def normalize(self, data, mean, var):
            return (data - mean) / np.sqrt(var)
    
    def denormalize(self, data, mean, var):
        return (data * np.sqrt(var)) + mean
    
    
    def predict_obs_los(self, dataset_dir, quasar):
                
        self.read_scalars_file()            
        print('scalars..', self.xscalar_mean, self.xscalar_var, 
              self.yscalar_mean, self.yscalar_var)
        
        x = self.uf.read_quasar_file(
            dataset_dir+quasar+'_z'+
            "{:.2f}".format(self.redshift)+'.npy')[0]

        if len(x.shape) == 1:
            x = np.expand_dims(np.expand_dims(x, axis=0), axis=2)
        else:
            x = np.expand_dims(x, axis=2)
        
        dist = self.ml_model(tf.convert_to_tensor(
            self.normalize(x, self.xscalar_mean, self.xscalar_var)), 
            training=False)
            
        mean = dist.mean()
        std = dist.stddev()

        upper_1sigma = mean + std
        lower_1sigma = mean - std

        x = np.squeeze(x, axis=2)
            
        mean = self.denormalize(mean, self.yscalar_mean, self.yscalar_var)
        upper_1sigma = self.denormalize(upper_1sigma, self.yscalar_mean, self.yscalar_var)
        lower_1sigma = self.denormalize(lower_1sigma, self.yscalar_mean, self.yscalar_var)
        
        
        print(self.output_quantity,'mean predictions', np.mean(mean), 
              np.mean(upper_1sigma), np.mean(lower_1sigma))            
        print(x.shape, mean.shape, std.shape)
            
        
        if x.shape[0] <= 10:
            sightlines_to_plot = x.shape[0]
        else:
            sightlines_to_plot = 10
        

        fig, ax = plt.subplots(sightlines_to_plot*2, 1, 
                                figsize=(28, 3*2*sightlines_to_plot))
        fig.subplots_adjust(wspace=0, hspace=0)
        
        axis = np.arange(x.shape[1]) /  (x.shape[1])
                
        for los in range(sightlines_to_plot):
            
            ax[los*2].step(axis, x[los], where='mid', linestyle='-', 
                            linewidth=2, color='black', alpha=1)
            ax[los*2].set_xlim(np.min(axis), np.max(axis))
            ax[los*2].set_ylabel(self.input_quantity)
            if self.input_quantity=='opt':
                ax[los*2].set_yscale('log')
            
            ax[los*2+1].step(axis, mean[los], where='mid', color='black', linestyle='--', 
                             linewidth=2, alpha=.6)
            ax[los*2+1].fill_between(axis, upper_1sigma[los], y2=lower_1sigma[los],
                                 color='red', alpha=.2)
            ax[los*2+1].set_xlim(np.min(axis), np.max(axis))
            ax[los*2+1].set_ylabel(self.output_quantity)
            if self.output_quantity=='opt':
                ax[los*2+1].set_yscale('log')


        fig.savefig(self.output_dir+quasar+'_'+self.input_quantity+'_'+
                    self.output_quantity+'.pdf',
                    format='pdf', dpi=90, bbox_inches = 'tight')
        plt.close()
        
        
        infilename = self.output_dir+'predict_'+quasar+'_'+self.input_quantity+'_'+\
        self.output_quantity+'.npy'
        print('saving ', infilename)

        with open(infilename, 'wb') as f:
            np.save(f, x)
            np.save(f, mean)
            np.save(f, upper_1sigma)
            np.save(f, lower_1sigma)
        
                   
    
    def predict(self, dpp):
                
        self.read_scalars_file()
        if self.input_quantity == "flux":
            self.x += self.n*np.random.normal(0, 1, np.shape(self.x))
           
        sightline_per_model = np.int32(self.x.shape[0] / len(self.files_list))
        
        print('x.shape..', self.x.shape)
        print('scalars..', self.xscalar_mean, self.xscalar_var, 
              self.yscalar_mean, self.yscalar_var)
        
        for mi, file in enumerate(self.files_list):
            print()
            x = self.x[mi*sightline_per_model:(mi+1)*sightline_per_model]
            y = self.y[mi*sightline_per_model:(mi+1)*sightline_per_model]
            
            dist = self.ml_model(tf.convert_to_tensor(
                self.normalize(x, self.xscalar_mean, self.xscalar_var)), 
                training=False)
            
            mean = dist.mean()
            std = dist.stddev()

            upper_1sigma = mean + std
            lower_1sigma = mean - std

            x = np.squeeze(x)
                
            mean = self.denormalize(mean, self.yscalar_mean, self.yscalar_var)
            upper_1sigma = self.denormalize(upper_1sigma, self.yscalar_mean, self.yscalar_var)
            lower_1sigma = self.denormalize(lower_1sigma, self.yscalar_mean, self.yscalar_var)
            
            
            print(self.output_quantity,'mean predictions', np.mean(mean), 
                  np.mean(upper_1sigma), np.mean(lower_1sigma))            
            print(mean.shape, std.shape)
                
            sightlines_to_plot = 10

            fig, ax = plt.subplots(sightlines_to_plot*2, 1, 
                                    figsize=(28, 3*2*sightlines_to_plot))
            fig.subplots_adjust(wspace=0, hspace=0)
            
            axis = np.arange(x.shape[1]) /  (x.shape[1])
            
            ax[0].text( 0.01,0.8, file.rstrip(".npy"), transform = ax[0].transAxes)

            
            for los in range(sightlines_to_plot):
                
                ax[los*2].step(axis, x[los], where='mid', linestyle='-', 
                                linewidth=2, color='black', alpha=1)
                ax[los*2].set_xlim(np.min(axis), np.max(axis))
                ax[los*2].set_ylabel(self.input_quantity)
                if self.input_quantity=='opt':
                    ax[los*2].set_yscale('log')
                
                ax[los*2+1].step(axis, y[los], where='mid', color='black', linestyle='-', 
                                linewidth=2, alpha=.6)
                
                ax[los*2+1].step(axis, mean[los], where='mid', color='black', linestyle='--', 
                                 linewidth=2, alpha=.6)
                ax[los*2+1].fill_between(axis, upper_1sigma[los], y2=lower_1sigma[los],
                                     color='red', alpha=.2)
                ax[los*2+1].set_xlim(np.min(axis), np.max(axis))
                ax[los*2+1].set_ylabel(self.output_quantity)
                if self.output_quantity=='opt':
                    ax[los*2+1].set_yscale('log')


            fig.savefig(self.output_dir+'los_'+self.input_quantity+'_'+
                        self.output_quantity+"_"+file.rstrip(".npy")+'.pdf',
                        format='pdf', dpi=90, 
                        bbox_inches = 'tight')
            plt.close()
            
            
            filename = self.output_dir+'predict_'+self.input_quantity+'_'+\
            self.output_quantity+'_'+file
            print('saving ', filename)

            with open(filename, 'wb') as f:
                np.save(f, x)
                np.save(f, y)
                np.save(f, mean)
                np.save(f, upper_1sigma)
                np.save(f, lower_1sigma)
