from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from scipy.ndimage import convolve1d
from typing import NoReturn, Union, Tuple, Any, Optional, List
from UtilityFunctions import UtilityFunctions

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

        
        self.bins_ratio = 1
        self.foreground_mean_flux = 1
        self.filename = ''
        self.post_file_name()
        self.uf  = UtilityFunctions()
        self.files_list  = self.uf.get_files_list_from_dir(
            self.dataset_dir, "dataset_",".npy")
        self.total_models = len(self.files_list)
    

    @property
    def vmax(self) -> float:
        return self.skewer_length / (1 + self.redshift) / self.hubble * self.hubbleZ
    
    @property
    def hubbleZ(self) -> float:
        return self.hubble*100.*np.sqrt(
            self.omegam*np.power(1.+self.redshift, 3) + (1.-self.omegam))

    def get_output_dir(self) -> str:
        return self.output_dir

    
    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                   np.ndarray, float, float]:
        return self.flux_dataset, self.densityw_dataset, \
    self.tempw_dataset, self.weights_dataset, self.scaler_mean, \
    self.scaler_var


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
    
    
    def read_skewers(self, filename) -> NoReturn:
        # check if the directory exists, and if not, create it
        if os.path.exists(self.dataset_dir):
            
            # Load the named arrays
            with open(self.dataset_dir+filename, 'rb') as f:
                loaded_data = np.load(f)
                        
                self.opt = loaded_data['opt']
                self.density = loaded_data['density']
                self.temp = loaded_data['temp']
                self.densityw = loaded_data['densityw']
                self.tempw = loaded_data['tempw']
                self.weights = loaded_data['weights']
                self.flux = np.exp(-self.opt)
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
                self.uf.downsample_field(flux_conv, self.flux_rebin, self.bins_ratio)
            else:
                self.uf.upsample_field(flux_conv, self.flux_rebin, self.bins_ratio)

            if iteration==0:
                  self.tempw_rebin = np.zeros((num_of_los, self.bins))
                  self.densityw_rebin = np.zeros((num_of_los, self.bins))
                
                  if bins_old>self.bins:
                      self.uf.downsample_field(self.densityw, self.densityw_rebin, self.bins_ratio)
                      self.uf.downsample_field(self.tempw, self.tempw_rebin, self.bins_ratio)
                  else:
                      self.uf.upsample_field(self.densityw, self.densityw_rebin, self.bins_ratio)
                      self.uf.upsample_field(self.tempw, self.tempw_rebin, self.bins_ratio)


                      
            fdiff =  self.mean_flux - np.mean(self.flux_rebin)
            iteration += 1
            self.mean_flux += fdiff
            print('<F> =', np.round(self.mean_flux, 3),  
                  np.round(np.mean(self.flux_rebin), 3), 
                  ', fdiff =', fdiff, ', bins =', self.bins)


    def save_processed_skewers(self, filename):
        
        save_file = self.output_dir+'/'+'processed_'+filename
        print(self.flux.shape, self.densityw.shape, self.tempw.shape)
        print('saving processed skewers file', save_file)
        with open(save_file, 'wb') as f:
            np.save(f, self.flux_rebin)
            np.save(f, self.densityw_rebin)
            np.save(f, self.tempw_rebin)

        
    
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

        
        self.scaler_mean = np.array([flux_scaler.mean_, 
                                     densityw_scaler.mean_, 
                                     tempw_scaler.mean_])


        self.scaler_var = np.array([flux_scaler.var_, 
                                     densityw_scaler.var_, 
                                     tempw_scaler.var_])
        
                
        save_file = self.output_dir+'scaler.npy'
        with open(save_file, 'wb') as f:
            np.save(f, self.scaler_mean)
            np.save(f, self.scaler_var)
            

    def stack_dataset(self) -> NoReturn:
        
        initialised = False
 
        for mi, filename in enumerate(self.files_list):
            
            print()
            print('reading/processing file', filename)
            
            self.filename = filename
            self.read_skewers(self.filename)
            self.process_skewers()
            self.save_processed_skewers(self.filename)
            
            # Check if weights are single valued for one model
            # repeate weights over all skewers
            if self.weights.size ==1:
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
        print('scalers', self.scaler_mean, self.scaler_var)
        print('datasets shapes ', self.flux_dataset.shape, 
              self.densityw_dataset.shape, 
              self.tempw_dataset.shape, self.weights_dataset.shape)
