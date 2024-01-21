import numpy as np
from typing import Union, Tuple, Any

class UtilityFunctions:
    
    def upsample(self, data, bins, bins_ratio):
        
        num_of_los = data.shape[0]
        bins_old = data.shape[1]
        data_rebin = np.zeros((num_of_los, bins))
        
        for ii in range(bins_old):
            data_rebin[:, np.int32((ii*bins_ratio)):np.int32(((ii+1)*bins_ratio))] = \
                    np.expand_dims(data[:, ii], axis=1)
         
        return data_rebin
    
    def downsample(self, data, bins, bins_ratio):
        
        num_of_los = data.shape[0]
        data_rebin = np.zeros((num_of_los, bins))
        
        for ii in range(bins):
            data_rebin[:, ii] = np.mean(
                data[:, np.int32((ii*bins_ratio)):np.int32(((ii+1)*bins_ratio))], axis=1)
         
        return data_rebin
        
    
    def rescale_opt(self, opt, mean_flux) -> float:
        #flatten the array
        opt_flt  = opt.flatten()

        #initialise with small value, so that the exp is a reasonable value
        a_pre = np.mean(np.exp(-opt_flt))
        diff = 1
            
        #use newton rapson method to get the factor
        while diff > 1e-8:
            num=(  np.mean( np.exp(-a_pre*opt_flt) * 1.0 ) - mean_flux  )            
            den=(  np.mean(opt_flt * np.exp(-a_pre*opt_flt)* 1.0 ) )          
            a_next=a_pre+num/den
            diff = np.abs(a_next - a_pre)
            a_pre=a_next
              
        return a_pre
    
    
    def hubblez(self, redshift, hubble, omegam):
        return hubble*100.*np.sqrt( omegam*(1.+redshift)*(1.+redshift)*(1.+redshift) + (1.-omegam))


    def fwhm_to_bins(self, fwhm, skewer_length, redshift, hubble, omegam):    
        sigma = fwhm/(2*np.sqrt(2 * np.log(2)))
        vmax = skewer_length/(1+redshift)/hubble*self.hubblez(redshift, hubble, omegam)
        return  np.int32(np.floor(vmax/sigma))


    def bins_to_fwhm(self, bins, skewer_length, redshift, hubble, omegam):    
        vmax = skewer_length/(1+redshift)/hubble*self.hubblez(redshift, hubble, omegam)
        return  vmax/np.float32(bins) * (2*np.sqrt(2 * np.log(2)))

    
    def closest_argmin(self, A, B):
        L = B.size
        sidx_B = B.argsort()
        sorted_B = B[sidx_B]
        sorted_idx = np.searchsorted(sorted_B, A)
        sorted_idx[sorted_idx==L] = L-1
        mask = (sorted_idx > 0) & \
        ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
        return sidx_B[sorted_idx-mask]

    
    def find_nearest(self, array: Union[list, np.ndarray], value: Any) -> Tuple[int, Any]:    
        
        array = np.asarray(array)
        index = (np.abs(array - value)).argmin()
        return index, array[index]
    
    
    def mean_flux_z(self, redshift: float = 4, filename: str = 'becker2013.dat') -> float:
            
        data = np.loadtxt(filename, usecols=(0, 5, 6), dtype=np.float32)
        redshift_index,_ = self.find_nearest(data[:,0], redshift)    
        mean_flux = data[redshift_index,1] #mean flux value for sigma
        
        # if redshift==5.0:
        #     mean_flux = 0.135
        # if redshift==5.2:
        #     mean_flux = 0.114
        
        return mean_flux
        
    