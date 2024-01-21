from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from scipy.ndimage import convolve1d
from typing import NoReturn, Tuple
from UtilityFunctions import UtilityFunctions


class DataProcessor:
    
    def __init__(
                 self,
        dataset_dir: str,
        dataset_file_filter: str,
        output_dir: str,
        input_quantity : str,
        output_quantity : str,
        noweights : bool,
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
        self.dataset_file_filter = dataset_file_filter
        self.output_dir = output_dir
        self.input_quantity = input_quantity
        self.output_quantity = output_quantity
        self.noweights = noweights
        
        self.redshift = redshift
        self.skewer_length = skewer_length
        self.hubble = hubble
        self.omegam = omegam
        self.fwhm = fwhm
        self.bins = bins
        self.mean_flux = mean_flux
        self.seed = seed
        self.uf = UtilityFunctions()
        
        self.xmean = 0
        self.xvar = 1
        self.index = []
        
        self.read_files_list()
        self.post_file_name()
    

    @property
    def vmax(self) -> float:
        return self.skewer_length / (1 + self.redshift) \
            / self.hubble * self.hubbleZ
    
    
    @property
    def hubbleZ(self) -> float:
        return self.hubble*100.*np.sqrt(
            self.omegam*np.power(1.+self.redshift, 3) + (1.-self.omegam))


    def get_output_dir(self) -> str:
        return self.output_dir
    
    
    def get_post_file_name(self) -> str:
        return self.post_output
    
    
    def get_files_list(self) -> str:
        return self.files_list
    
    
    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                   np.ndarray, float, float]:
        return self.xdataset, self.ydataset, self.wdataset, \
    self.xmean, self.xvar, self.index


    def read_files_list(self) -> NoReturn:
        if os.path.exists(self.dataset_dir):
        
            file_list = os.listdir(self.dataset_dir)
            filtered_files = [filename for filename in file_list
                              if self.dataset_file_filter in filename]
            self.files_list = sorted(filtered_files)            
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
        # check if the directory exists, and if not, raise error
        filename = self.dataset_dir+self.filename
        
        if os.path.exists(filename):
            data = np.load(filename,'rb')
            
            if self.input_quantity in data:
                
                self.x = data[self.input_quantity]
                
                if self.input_quantity == 'opt':
                    #if optical depth rescale them to match a mean flux
                    self.x  = self.uf.rescale_opt(self.x, self.mean_flux)*self.x
                    
                # rebin data to pixels specified
                self.x = self.rebin(self.x)
                    
            elif self.input_quantity == "flux" and "opt" in data:
                # this function returns flux
                self.x = self.process_opt(data["opt"])
            else:    
                print('The input', self.input_quantity ,'does not exist')


            if self.output_quantity in data:
                
                self.y = data[self.output_quantity]
                
                if self.output_quantity == 'opt':
                    self.y  = self.uf.rescale_opt(self.y, self.mean_flux)*self.y
                    
                # rebin data to pixels specified
                self.y = self.rebin(self.y)
                    
            elif self.output_quantity == "flux" and "opt" in data:
                # this function returns flux
                self.y = self.process_opt(data["opt"])
            else:    
                print('The input', self.output_quantity ,'does not exist')

            
            if 'weights' in data and not self.noweights:
                self.w = self.rebin(data["weights"])
            else:
                self.w = np.ones(self.x.shape)
                
            print('input', self.input_quantity, self.x.shape, np.mean(self.x))
            print('output', self.output_quantity, self.y.shape, np.mean(self.y))
            print("weights", self.w.shape, np.mean(self.w))
                  
        else:
            raise ValueError('file: {filename} doest not exist' )


    def rebin(self, data):
        
        bins_old = data.shape[1]
        
        bins_ratio = (bins_old/self.bins) if bins_old>self.bins else (self.bins/bins_old)
        
        if bins_old>self.bins:
            data_rebin = self.uf.downsample(data, self.bins, bins_ratio)
        elif bins_old<self.bins:
            data_rebin = self.uf.upsample(data, self.bins, bins_ratio)
        
        return data_rebin
      

    def convolve(self, data):
        
        bins_old = data.shape[1]
        num_of_los = data.shape[0]

        fwhmrel = self.fwhm * bins_old/self.vmax
        sigma = fwhmrel/(2*np.sqrt(2*np.log(2)))

        #roughly 50 pixels have non-zero values in kernel
        kernel_bins = 50 #flux.shape[1]
        kernel_bins_half = np.int32(kernel_bins/2)

        xx = np.arange(kernel_bins) - kernel_bins_half 
        kernel = 1/(sigma*np.sqrt(2*np.pi))  * np.exp(-0.5 * (xx/sigma)*(xx/sigma))            
        data_conv = np.zeros(data.shape)
            
        for ilos in range(num_of_los):
            data_conv[ilos,:] = convolve1d(data[ilos,:], kernel, mode='wrap')

        return data_conv


    def process_opt(self, opt):
        
        iteration = 0
        fdiff = 1
        iterations_allowed = 1

        while np.abs(fdiff)>1e-4 and iteration<iterations_allowed:
            
            #STEP 1: Rescale flux
            flux = np.exp(-self.uf.rescale_opt(opt, self.mean_flux)*opt)
            
            #STEP 2: Convolve with Gaussian profile
            #sigma = 1 means it unchanged
            flux_conv = self.convolve(flux)

            #STEP 3: Rebin onto pixels
            #pixel size to sigma (based on FWHM) width sigma = FWHM/[2*(2*ln2)^1/2]  
            flux_rebin = self.rebin(flux_conv)

            current_mean_flux = np.mean(flux_rebin)
            fdiff =  self.mean_flux - current_mean_flux
            iteration += 1
            self.mean_flux += fdiff
            print('<F> difference =', fdiff, ', bins =', self.bins)
        return flux_rebin



    def scale_dataset(self) -> NoReturn:
                
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        
        #normalize the dataset
        xscaler.fit(self.xdataset.reshape(-1, 1))
        yscaler.fit(self.ydataset.reshape(-1, 1))
    
        print()
        print('scalers..')
        print(self.input_quantity, xscaler.mean_, xscaler.var_)
        print(self.output_quantity, yscaler.mean_, yscaler.var_)
        
        self.xdataset = xscaler.transform(
            self.xdataset.reshape(-1,1)).reshape(self.xdataset.shape)
        self.ydataset = yscaler.transform(
            self.ydataset.reshape(-1,1)).reshape(self.ydataset.shape)
        
        self.xmean = xscaler.mean_
        self.xvar = xscaler.var_
                
        save_file = self.output_dir+'scaler_'+self.input_quantity+'_'+self.output_quantity
        with open(save_file, 'wb') as f:
            np.save(f, xscaler.mean_)
            np.save(f, xscaler.var_)
            np.save(f, yscaler.mean_)
            np.save(f, yscaler.var_)
      

    def stack_dataset(self) -> NoReturn:
        
        initialised = False

        for mi, filename in enumerate(self.files_list):
            
            print()
            print('reading/processing file', filename)
            
            self.filename = filename
            self.read_skewers()
                
            if initialised!=True:
                initialised = True
                self.xdataset = self.x
                self.ydataset = self.y
                self.wdataset = self.w
            else:
                self.xdataset = np.vstack( (self.xdataset, self.x) )
                self.ydataset = np.vstack( (self.ydataset, self.y) )
                self.wdataset = np.vstack( (self.wdataset, self.w) )

            self.index = np.arange(self.xdataset.shape[0])


    def shuffle_dataset(self) -> NoReturn:
        
        sightline_per_model = np.int32(self.xdataset.shape[0] / self.total_models )
        index = np.zeros(self.xdataset.shape[0])
        
        for i in range(len(index)):
            index[i] = (i%self.total_models) * sightline_per_model + \
            np.int32(i/self.total_models)
            
        self.index = index.astype('int32')
        self.xdataset = self.xdataset[self.index]
        self.ydataset = self.ydataset[self.index]
        self.wdataset = self.wdataset[self.index]


    def make_dataset(self, scale_and_shuffle) -> NoReturn:
 
        self.stack_dataset()
        print()
        
        print(self.input_quantity, self.output_quantity, 'before scaling mean ', 
              np.mean(self.xdataset), np.mean(self.ydataset), np.mean(self.wdataset))
        
        if scale_and_shuffle:
            self.scale_dataset()
            self.shuffle_dataset()
            
        print(self.input_quantity, self.output_quantity, 'weights', 'after scaling mean ', 
                  np.mean(self.xdataset), np.mean(self.ydataset), np.mean(self.wdataset))
            
        print(self.input_quantity, self.output_quantity, 'weights', 'shapes', 
                  self.xdataset.shape, self.ydataset.shape, self.wdataset.shape)
