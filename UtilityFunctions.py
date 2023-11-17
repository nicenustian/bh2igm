import numpy as np
import os
from typing import Union, Tuple, Any, Optional, List


class UtilityFunctions:
    
    def upsample_field(self, field, field_rebin, bins_ratio):
        
        bins = field.shape[1]
        
        # Ensure that the second axis of field_rebin is larger than the original
        assert field_rebin.shape[1] > field.shape[1], "Second axis must be larger than the first axis."
        
        #upsampling
        for ii in range(bins):
            field_rebin[:, np.int32((ii*bins_ratio)):np.int32(((ii+1)*bins_ratio))] = \
                np.expand_dims(field[:, ii], axis=1)

         
    
    def downsample_field(self, field, field_rebin, bins_ratio):
        
        # Ensure that the second axis of field is larger than the rebinned
        assert field_rebin.shape[1] < field.shape[1], "Second axis must be smaller than the original."
        
        
        bins = field_rebin.shape[1]

        for ii in range(bins):
            field_rebin[:, ii] = np.mean(
                field[:, np.int32((ii*bins_ratio)):np.int32(((ii+1)*bins_ratio))], axis=1
                )
         
        
    
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
        return mean_flux
    
    
    def get_files_list_from_dir(self, dataset_dir: str, filter_start: str, 
                                filter_end: str) -> List[str]:
        if os.path.exists(dataset_dir):
        
            file_list = os.listdir(dataset_dir)
            # Filter files with string in the filenames or remove any hidden files
            filtered_files = [filename for filename in file_list if 
                              (filename.startswith(filter_start) and 
                            filename.endswith(filter_end))]
            files_list = filtered_files
        else:
            raise ValueError('directory: {self.dataset_dir} doest not exist' )
    
        return files_list
        
        
    