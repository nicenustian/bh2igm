from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from scipy.ndimage import convolve1d
from typing import NoReturn, Union, Tuple, Any, Optional

class DataProcessor:
    
    def __init__(
                 self,
        dataset_dir: str,
        output_dir: str,
        redshift: float,
        skewer_length: int,
        hubble: float,
        omegam: float,
        fwhm: float,
        bins: int,
        mean_flux: float,
        seed: int
    ) -> None:
        
        
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        
        self.redshift = redshift
        self.skewer_length = skewer_length
        self.hubble = hubble
        self.omegam = omegam
        self.fwhm = fwhm
        self.bins = bins
        self.mean_flux = mean_flux
        self.seed = seed
        
        self.opt = np.array([])
        self.flux = np.array([])
        self.flux_rebin = np.array([])
        
        self.density = np.array([])
        self.temp = np.array([])
        self.densityw = np.array([])
        self.tempw = np.array([])
        self.weights = np.array([])
        
        self.densityw_rebin = np.array([])
        self.tempw_rebin = np.array([])
        
        self.flux_dataset = np.array([])
        self.densityw_dataset = np.array([])
        self.tempw_dataset = np.array([])
        self.weights_dataset = np.array([])
        
        self.flux_scaler_mean = 1
        self.flux_scaler_var = 1
        
        self.bins_ratio = 1
        self.foreground_mean_flux = 1
        self.filename = ''
        self.get_files_list()
        self.post_file_name()
    

    @property
    def vmax(self) -> float:
        return self.skewer_length / (1 + self.redshift) / self.hubble * self.hubbleZ
    
    @property
    def hubbleZ(self) -> float:
        return self.hubble*100.*np.sqrt(self.omegam*np.power(1.+self.redshift, 3) + (1.-self.omegam))

    def get_output_dir(self) -> str:
        return self.output_dir
    
    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                   np.ndarray, float, float]:
        return self.flux_dataset, self.densityw_dataset, \
    self.tempw_dataset, self.weights_dataset, self.flux_scaler_mean, \
    self.flux_scaler_var


    def get_files_list(self) -> NoReturn:
        if os.path.exists(self.dataset_dir):
        
            file_list = os.listdir(self.dataset_dir)
            # Filter files that do not end with ".npy"
            filtered_files = [filename for filename in file_list if filename.endswith(".npy")]
            self.files_list = filtered_files

            self.total_models = len(self.files_list)
        else:
            raise ValueError('directory: {self.dataset_dir} doest not exist' )
    
        return 
        

    def post_file_name(self) -> NoReturn:
        self.post_output = '_mflux'+"{:.4f}".format(self.mean_flux)+\
        '_fwhm'+"{:.2f}".format(self.fwhm)+'_z'+"{:.2f}".format(self.redshift)

        self.output_dir = self.output_dir.replace("/", "")+self.post_output+"/"
            
        # check if the directory exists, and if not, create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Directory '{self.output_dir}' created successfully.")
        else:
            print(f"Directory '{self.output_dir}' already exists.")
    
    
    def read_skewers(self) -> NoReturn:
        # check if the directory exists, and if not, create it
        if os.path.exists(self.dataset_dir):
            
            # Load the named arrays
            with open(self.dataset_dir+self.filename, 'rb') as f:
                loaded_data = np.load(f)
                        
                self.opt = loaded_data['opt']
                self.density = loaded_data['density']
                self.temp = loaded_data['temp']
                self.densityw = loaded_data['densityw']
                self.tempw = loaded_data['tempw']
                self.weights = loaded_data['weights']
        else:
            raise ValueError('directory: {self.dataset_dir} doest not exist' )
    
    
    def rescale_tau(self) -> float:
        #flatten the array
        opt_flt  = self.opt.flatten()

        #initialise with small value, so that the exp is a reasonable value
        a_pre = np.mean(np.exp(-self.opt))
        diff = 1
            
        #use newton rapson method to get the factor
        while diff > 1e-8:
            num=(  np.mean( np.exp(-a_pre*opt_flt) * self.foreground_mean_flux ) - self.mean_flux  )            
            den=(  np.mean(opt_flt * np.exp(-a_pre*opt_flt)*self.foreground_mean_flux ) )          
            a_next=a_pre+num/den
            diff = np.abs(a_next - a_pre)
            a_pre=a_next
              
        return a_pre


    def process_skewers(self) -> NoReturn:
        
        iteration = 0
        fdiff = 1
        iterations_allowed = 1
        
        bins_old = self.opt.shape[1]
        num_of_los = self.opt.shape[0]

        while np.abs(fdiff)>1e-4 and iteration<iterations_allowed:
            
            #STEP 1: Rescale flux
            flux = np.exp(-self.rescale_tau()*self.opt)
            
            #STEP 2: Convolve with Gaussian profile
            #sigma = 1 means it unchanged
            FWHMrel = self.fwhm * bins_old/self.vmax
            sigma = FWHMrel/(2*np.sqrt(2*np.log(2)))

            #roughly 50 pixels have non-zero values in kernel
            kernel_bins = 50 #flux.shape[1]
            kernel_bins_half = np.int32(kernel_bins/2)

            xx = np.arange(kernel_bins) - kernel_bins_half 
            kernel = 1/(sigma*np.sqrt(2*np.pi))  * np.exp(-0.5 * (xx/sigma)*(xx/sigma))            
            flux_conv = np.zeros(flux.shape)
            
            for ilos in range(num_of_los):
                flux_conv[ilos,:] = convolve1d(flux[ilos,:], kernel, mode='wrap')

            #STEP 3: Rebin onto pixels
            #######################################################################
            #pixel size to sigma
            #of width sigma = fwhm/[2*(2*ln2)^1/2] 

            self.flux_rebin = np.zeros((num_of_los, self.bins))
            self.bins_ratio = (bins_old/self.bins) if bins_old>self.bins else (self.bins/bins_old)
               
            if bins_old>self.bins:
                 #down sampling
                 for ii in range(self.bins):
                         self.flux_rebin[:, ii] = np.mean(
                                 flux_conv[:, np.int32(
                                     (ii*self.bins_ratio)):np.int32(((ii+1)*self.bins_ratio))], axis=1)
         
            else:
                #upsampling
                for ii in range(bins_old):
                    self.flux_rebin[:, np.int32(
                        (ii*self.bins_ratio)):np.int32(((ii+1)*self.bins_ratio))] = \
                    np.expand_dims(flux_conv[:, ii], axis=1)


            if iteration==0:
                  self.tempw_rebin = np.zeros((num_of_los, self.bins))
                  self.densityw_rebin = np.zeros((num_of_los, self.bins))
                
                  if bins_old>self.bins:
                      for ii in range(self.bins):
                              self.tempw_rebin[:, ii] = np.mean(
                                  self.tempw[:, np.int32((ii*self.bins_ratio)):np.int32(
                                      ((ii+1)*self.bins_ratio))], axis=1)
                         
                              self.densityw_rebin[:, ii] = np.mean(
                                  self.densityw[:, np.int32((ii*self.bins_ratio)):np.int32(
                                      ((ii+1)*self.bins_ratio))], axis=1)
                             
                  else:
                      for ii in range(bins_old):
                          self.densityw_rebin[:, np.int32(np.round(ii*self.bins_ratio)):np.int32(
                              np.round((ii+1)*self.bins_ratio))] = np.expand_dims(self.densityw[:, ii], axis=1)
                          self.tempw_rebin[:, np.int32(np.round(ii*self.bins_ratio)):np.int32(
                              np.round((ii+1)*self.bins_ratio))] = np.expand_dims(self.tempw[:, ii], axis=1)

            fdiff =  self.mean_flux - np.mean(self.flux_rebin)
            iteration += 1
            self.mean_flux += fdiff
            print('<F> =', np.round(self.mean_flux, 3),  np.round(np.mean(self.flux_rebin), 3), 
                  ', fdiff =', fdiff, ', bins =', self.bins)


    def scale_dataset(self) -> NoReturn:
                
        flux_scaler = StandardScaler()
        densityw_scaler = StandardScaler()
        tempw_scaler = StandardScaler()
        
        #normalize the dataset
        flux_scaler.fit(self.flux_dataset.reshape(-1, 1))
        densityw_scaler.fit(self.densityw_dataset.reshape(-1, 1))
        tempw_scaler.fit(self.tempw_dataset.reshape(-1, 1))
    
        print('scalers=', flux_scaler.mean_, flux_scaler.var_, 
              densityw_scaler.mean_, densityw_scaler.var_,
              tempw_scaler.mean_, tempw_scaler.var_)
        
        
        self.flux_dataset = flux_scaler.transform(
            self.flux_dataset.reshape(-1,1)).reshape(self.flux_dataset.shape)
        self.densityw_dataset = densityw_scaler.transform(
            self.densityw_dataset.reshape(-1,1)).reshape(self.densityw_dataset.shape)
        self.tempw_dataset = tempw_scaler.transform(
            self.tempw_dataset.reshape(-1,1)).reshape(self.tempw_dataset.shape)
        
        self.flux_scaler_mean = flux_scaler.mean_
        self.flux_scaler_var = flux_scaler.var_
        
                
        save_file = self.output_dir+'scaler'+self.post_output
        with open(save_file, 'wb') as f:
            np.save(f, flux_scaler.mean_)
            np.save(f, flux_scaler.var_)
            np.save(f, densityw_scaler.mean_)
            np.save(f, densityw_scaler.var_)
            np.save(f, tempw_scaler.mean_)
            np.save(f, tempw_scaler.var_)


    def stack_dataset(self) -> NoReturn:
        
        initialised = False
 
        for mi, filename in enumerate(self.files_list):
            
            print()
            print('reading/processing file', filename)
            
            self.filename = filename
            self.read_skewers()
            self.process_skewers()
            
            #print(self.weights)
            # Check if weights are single valued for one model
            # repeate weights over all skewers
            #if isinstance(self.weights, float):
            self.weights = np.full(self.flux_rebin.shape, self.weights)

            
            if initialised!=True:
                initialised = True
                self.flux_dataset = self.flux_rebin
                self.densityw_dataset = self.densityw_rebin
                self.tempw_dataset = self.tempw_rebin
                self.weights_dataset = self.weights
            else:                
                self.flux_dataset = np.vstack( (self.flux_dataset, self.flux_rebin) )
                self.densityw_dataset = np.vstack( (self.densityw_dataset, self.densityw_rebin) )
                self.tempw_dataset = np.vstack( (self.tempw_dataset, self.tempw_rebin) )
                self.weights_dataset = np.vstack( (self.weights_dataset, self.weights) )


    def shuffle_dataset(self) -> NoReturn:
        sightline_per_model = np.int32(self.flux_dataset.shape[0] / self.total_models )
        index = np.zeros(self.flux_dataset.shape[0])
        for i in range(len(index)):
            index[i] = (i%self.total_models) * sightline_per_model + \
            np.int32(i/self.total_models)
            
                
        index = index.astype('int32')
        self.flux_dataset = self.flux_dataset[index]
        self.densityw_dataset = self.densityw_dataset[index]
        self.tempw_dataset = self.tempw_dataset[index]
        self.weights_dataset = self.weights_dataset[index]


    def make_dataset(self) -> NoReturn:
 
        self.stack_dataset()
        print()
        
        print('before scaling mean (flux, densityw, tempw, weights) ', 
              np.mean(self.flux_dataset), np.mean(self.densityw_dataset),
              np.mean(self.tempw_dataset), np.mean(self.weights_dataset))
        
        self.scale_dataset()
        self.shuffle_dataset()            

        print('after scaling mean (flux, densityw, tempw, weights)', 
              np.mean(self.flux_dataset), np.mean(self.densityw_dataset), 
              np.mean(self.tempw_dataset))
        print('datasets shapes ', self.flux_dataset.shape, 
              self.densityw_dataset.shape, 
              self.tempw_dataset.shape, self.weights_dataset.shape)
