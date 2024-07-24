import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from adds import correlation_from_covariance, read_predictions


font = {'family' : 'serif', 'weight' : 'normal','size' : 34}
matplotlib.rc('font', **font)
cl = ['red', 'blue', 'orange', 'purple', 'cyan', 'grey',
      'green','magenta', 'tan', 'green','magenta', 'tan', 
      'green','magenta', 'tan']

lw = 2
skewer_length = 20
trim = 16

dir_dataset = 'bh2igm/dataset_files/'
dir_output = 'bh2igm/'
# dataset_file_filter = 'ml_outputs_J'

dir_output = 'bh2igm/ml_outputs_mflux0.1350_fwhm6.00_bins1067_noise0.02_z5.00/'
#dir_output = 'bh2igm/ml_outputs_mflux0.3216_fwhm6.00_bins1014_noise0.02_z4.40/'
#dir_output = 'bh2igm/ml_outputs_trim16_mflux0.4255_fwhm6.00_bins978_noise0.02_z4.00/'
#dir_dataset = 'bh2igm/dataset_files/'

models = [
    'planck1_20_1024',       'planck1_20_1024_cold',        'planck1_20_1024_hot',
    'planck1_20_1024_zr525', 'planck1_20_1024_zr525_cold',  'planck1_20_1024_zr525_hot',
    'planck1_20_1024_zr675', 'planck1_20_1024_zr675_cold',  'planck1_20_1024_zr675_hot',
    'planck1_20_1024_zr750', 'planck1_20_1024_zr750_cold',  'planck1_20_1024_zr750_hot',
    'planck1_20_1024_g10',   'planck1_20_1024_g14',         'planck1_20_1024_g16',
]

redshifts = [4.0]

#models = ["nyx_zre6", "nyx_zre7", "nyx_zre7_hot", "nyx_zre7_cold", "nyx_zre8"]

###############################################################################

def concat_models(densityw_res, tempw_res, pixels, realizations=40):
    
    intialization = False
    for ri in range(realizations):
        si = np.random.randint(0, pixels)
        concat =  np.hstack((
            np.roll(densityw_res, si, axis=1), np.roll(tempw_res, si, axis=1)))
        
        if intialization==False:
            intialization=True
            concat_dataset = concat
        else:
            concat_dataset = np.vstack((concat_dataset, concat))

    return concat_dataset


def get_cov(dir_dataset, dir_output, model, redshift):
    
    flux,_,_, densityw, tempw, densityw_mean, densityw_upper_1sigma , \
    densityw_lower_1sigma, tempw_mean, tempw_upper_1sigma , tempw_lower_1sigma  = \
        read_predictions(dir_dataset, dir_output, model, redshift)
    
                
    # ##remove trimmed pixels
    # tempw = tempw[:,trim:]
    # densityw = densityw[:,trim:]
    # flux = flux[:,trim:]

    densityw_std = densityw_upper_1sigma - densityw_mean
    tempw_std = tempw_upper_1sigma - tempw_mean
    
    pixels = densityw_std.shape[1]
    pixelsh = np.int32(pixels/2)
    
    start_time = time.time()
    
    #X  - E(X)  = covaraince
    densityw_res = (densityw - densityw_mean)/densityw_std
    tempw_res = (tempw - tempw_mean)/tempw_std
        
    concat_dataset = concat_models(densityw_res, tempw_res, pixels, realizations=40)
    print('sec ',time.time() - start_time, concat_dataset.shape)
                        
    corr = np.cov(concat_dataset, rowvar=False)
    corr = correlation_from_covariance(corr)
    
    print('corr', np.min(corr), np.max(corr), corr.shape)

    save_file = dir_output+'corr_'+model+'.npy'
    print('saving ', save_file)
    with open(save_file, 'wb') as f:
        np.save(f, corr)
        
    return corr


# file_list = os.listdir(dir_output)
# dir_output_folder_list = [filename for filename in file_list
#                   if dataset_file_filter in filename]


# for file_folder in dir_output_folder_list:
#     for mi, model in enumerate(models):
    
#         string_split = (file_folder).split('_')
    
#         quasar = string_split[2]
#         redshift = np.float32(string_split[7][1:])

for zi, redshift in enumerate(redshifts):
    for mi, model in enumerate(models):
        
        corr = get_cov(dir_dataset, dir_output, model, redshift)
                
###########################################################################
                
        axis = np.arange(np.int32(corr.shape[1]/2)) * (skewer_length/np.int32(corr.shape[1]/2))
        axish = np.arange(np.int32(corr.shape[1]/4)) * (skewer_length/np.int32(corr.shape[1]/4))
        half_pixels = np.int32(corr.shape[1]/2)
        
        # Where we want the ticks, in pixel locations
        ticks = np.linspace(0, corr.shape[1]/2, 5)
        # What those pixel locations correspond to in data coordinates.
        # Also set the float format here
        ticklabels = [i for i in np.int32(ticks*skewer_length/np.int32(corr.shape[1]/2))]
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        fig.subplots_adjust(wspace=0.001, hspace=0.001)
       
        ax[0,0].imshow(corr[:half_pixels, :half_pixels], vmin=0, vmax=1, label=model)
        ax[0,0].set_xticklabels([])
        ax[0,0].set_yticks(ticks)
        ax[0,0].set_yticklabels(ticklabels)
        ax[0,0].set_ylabel(r'${\rm Mpc/h}$')
        ax[0,0].set_aspect("auto")
        

        cbar = ax[0,1].imshow(corr[:half_pixels, half_pixels:], vmin=0, vmax=1, label=model)
        ax[0,1].set_xticklabels([])
        ax[0,1].set_yticklabels([])
        ax[0,1].set_aspect("auto")
        cax = fig.add_axes([.95, .2, 0.02, .6])
        fig.colorbar(cbar, orientation='vertical', cax=cax)

        
        ax[1,0].imshow(corr[half_pixels:, :half_pixels], vmin=0, vmax=1, label=model)
        ax[1,0].set_xticks(ticks)
        ax[1,0].set_xticklabels(ticklabels)
        ticklabels[0] = ''
        ax[1,0].set_yticks(ticks)
        ax[1,0].set_yticklabels(ticklabels)
        ax[1,0].set_xlabel(r'${\rm Mpc/h}$')
        ax[1,0].set_ylabel(r'${\rm Mpc/h}$')
        ax[1,0].set_aspect("auto")
        
        
        ax[1,1].imshow(corr[half_pixels:, half_pixels:], vmin=0, vmax=1, label=model)
        ax[1,1].set_yticklabels([])
        ax[1,1].set_xticks(ticks)
        ticklabels[0] = ''
        ax[1,1].set_xticklabels(ticklabels)
        ax[1,1].set_xlabel(r'${\rm Mpc/h}$')
        ax[1,1].set_aspect("auto")
                  
        fig.savefig('corr_'+model+'.pdf', format='pdf', dpi=90, bbox_inches = 'tight')
