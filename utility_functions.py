import numpy as np
from typing import Union, Tuple, Any, Optional


class UtilityFunctions:
    
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
        
    