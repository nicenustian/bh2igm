import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from adds import t0_gamma_from_bs_corr, get_contours
from adds import read_predictions, t0_gamma_from_bs, get_t0_gamma_values_str
from adds import t0_gamma_to_sim_model, findNearest
from adds import find_conf_int,  project_temp_on_delta, t0_gamma
from scipy import stats
import os
from cov_models import get_cov

font = {'family' : 'serif', 'weight' : 'normal','size' : 34}
matplotlib.rc('font', **font)
#cl = ['red', 'blue', 'orange','purple', 'olive', 'tan', 'green','magenta']

lw = 2
seed = 12345
np.random.seed(seed)
dir_output = 'ml_outputs/'
skewer_length = 20
hubble = 0.678
realizations = 1000
los = 0
trim = 16

dir_dataset = 'bh2igm/dataset_files/'
dir_output = 'bh2igm/'
dataset_file_filter = 'ml_outputs_J'

file_list = os.listdir(dir_output)
dir_output_folder_list = [filename for filename in file_list
                  if dataset_file_filter in filename]


redshifts = [3.96, 4.0, 4.04,
             4.20, 4.24, 4.35, 
             4.4, 4.45,  4.55, 
             4.6, 4.65,  
             4.75, 4.8]

quasar_list = ['J021043', 'J025019', 'J145147','J120523', 'J004054', 'J025019','J030722', 'J033829']
cl = ['red', 'blue', 'orange','purple', 'olive', 'tan', 'green','magenta']

###############################################################################

def get_cont(t0, gamma):

    t0 = t0.flatten()
    gamma = gamma.flatten()
    
    xmin = -2
    xmax = 3
    ymin = 3
    ymax = 5
            
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # select random samples
    index = np.random.randint(0, len(t0), 1000)
    values = np.vstack([t0[index], gamma[index]])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    conf = find_conf_int(Z)
    
    return X,Y,Z, conf


###############################################################################

if len(redshifts)==1:
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
else:
    fig, ax = plt.subplots(1, len(redshifts), figsize=(10*len(redshifts), 8))
fig.subplots_adjust(wspace=0.0, hspace=.0)

for file_folder in dir_output_folder_list:
    
    string_split = (file_folder).split('_')
    
    quasar = string_split[2]
    redshift = np.float32(string_split[7][1:])
    
    zi = findNearest(redshifts, redshift)[0]
    qso_index = quasar_list.index(quasar)
    qs_color = cl[qso_index]
    
    filename  = dir_dataset +"/"+ quasar + '_z'+"{:.2f}".format(redshift) + ".npy"
        
    data = np.load(filename,'rb')
    flux = data["flux_obs"]
    # update mean for flux, fwhm and noise
    mean_flux = data["mean_flux"]
    fwhm = data["fwhm"]
    noise = data["noise"]
    bins = data["bins"]
    flux_level = data["flux_level"]
    noise_level = data["noise_level"]
    noise_obs = data["noise_obs"]
    snr = 1/np.mean(noise[noise!=0])

    if snr>=100:

        initialized = False
        combined_real = list()
        combined_mask = list()
        initialized = False
    
        _, densityw_mean, densityw_upper_1sigma , densityw_lower_1sigma, \
        tempw_mean, tempw_upper_1sigma , tempw_lower_1sigma  = \
        read_predictions(dir_dataset+"/", dir_output+"/"+file_folder+"/", 
                             quasar, redshift, False
                             )
        
        
        densityw_std = densityw_upper_1sigma - densityw_mean
        tempw_std = tempw_upper_1sigma - tempw_mean
        
        mean_concat = np.hstack((densityw_mean, tempw_mean))
        std_concat = np.hstack((densityw_std, tempw_std))
        
        # remove trimmed pixels
        flux = flux[trim:]
        noise_obs = noise_obs[trim:]
        
        t0 = np.full(realizations, np.nan)
        gamma = np.full(realizations, np.nan)
    
        #mask for valid flux pixels
        mask = np.logical_or(flux > noise_obs, flux!=1)
        
        t0_noncorr, gamma_noncorr = t0_gamma_from_bs(
                    densityw_mean[los][mask[los]], densityw_std[los][mask[los]], 
                    tempw_mean[los][mask[los]], tempw_std[los][mask[los]])
        
        
        t0m, gammam, tempstr, gammastr = get_t0_gamma_values_str(
            t0_noncorr, gamma_noncorr
            )
        
        print('without corr = ', tempstr, gammastr)

        
        sim_model = t0_gamma_to_sim_model(dir_dataset,
              np.median(t0_noncorr), np.median(gamma_noncorr), redshift)
        
            
        filename = dir_output+"/"+file_folder+"/"+'corr_'+sim_model+'.npy'
        
        if os.path.isfile(filename):
            corr = np.load(filename)
        else:
            corr = get_cov(dir_dataset+"/", dir_output+"/"+file_folder+"/", sim_model, redshift)
            
        
        pixels = densityw_mean.shape[1]
        concat_real_orig = np.random.multivariate_normal(
                np.zeros(pixels*2), corr, realizations
                )
        
        concat_real = mean_concat[los] + concat_real_orig * std_concat[los]
            
        t0 = np.full(realizations, np.nan)
        gamma = np.full(realizations, np.nan)
            
        for ri in range(realizations):
            t0[ri], gamma[ri] = t0_gamma(concat_real[ri, :pixels][mask], 
                             concat_real[ri, pixels:][mask])
            
        crl, crm, cru = np.quantile(concat_real, [0.16, 0.5, 0.84], axis=0)
        
        t0m, gammam, tempstr, gammastr = get_t0_gamma_values_str(t0, gamma)                
        print('with corr = ', tempstr, gammastr)
        #print(zi, qso_index, tempstr, gammastr, qs_color)
            
        #ax[zi].text(0.04, 0.92-0.08*qso_index, r'$T_{\rm 0}$=' + tempstr+r'$\gamma=$'+gammastr,
        #fontsize=24, transform = ax[zi].transAxes, color=qs_color)
        
        ax[zi].text(0.04,0.04+qso_index*0.06, 
                      quasar+r', $T_{\rm 0}$=' +\
                          str(int(np.mean(10**t0)))+r'${\rm K}$' + ', ' + r'$\gamma=$'\
                              +str(np.round(np.mean(gamma), 2))\
                          +', '+r'${\rm S/N}=$'+str(int(snr))+\
                              ', '+r'${\rm \langle F \rangle}=$'+\
                                  str(np.round(np.mean(flux[mask]), 2)),
                      fontsize=20, transform = ax[zi].transAxes, color=qs_color)
    
                     
        concat_mask = np.repeat(mask[np.newaxis, :], realizations, axis=0)
        
        X, Y, Z, conf = get_cont(concat_real[:,:pixels][concat_mask], 
            concat_real[:,pixels:][concat_mask])
                    
        density_pro, temp_pro = project_temp_on_delta(
            densityw_mean, tempw_mean, delta_bin=0.04)
                    
        # the actual model quantities
        #ax[zi].pcolormesh(X, Y, Z)
        ax[zi].contour(X, Y, Z, [conf[1]], colors=qs_color, linestyles='solid')
        ax[zi].plot(density_pro, temp_pro, linewidth=2, linestyle='--', 
                    color=qs_color)
    
    
        if initialized:
            combined_real = np.vstack((combined_real, concat_real))
            combined_mask = np.vstack((combined_mask, concat_mask))
    
        else:
            initialized = True
            combined_real = concat_real
            combined_mask = concat_mask
            
        #######################################################################          
    
        X,Y,Z,conf = get_cont(combined_real[:,:pixels][combined_mask], 
              combined_real[:,pixels:][combined_mask])
                    
        density_pro, temp_pro = project_temp_on_delta(
            combined_real[:,:pixels][combined_mask],
            combined_real[:,pixels:][combined_mask], 
            delta_bin=0.04)
        
        t0m, gammam, tempstr, gammastr = get_t0_gamma_values_str(t0, gamma)                
        print('with corr = ', tempstr, gammastr)
        
        # the actual model quantities
        #ax[zi].pcolormesh(X, Y, Z)
        ax[zi].contour(X, Y, Z, conf, colors='white', linestyles='solid')
        ax[zi].plot(density_pro, temp_pro, linewidth=2, linestyle='--', color='white')
        ax[zi].set_xlim(-1.2, .4)
        ax[zi].set_ylim(3.3, 4.5)
        
        ax[zi].text(
            0.7,0.04, r'$z=$'+"{:.2f}".format(redshift), fontsize=28, 
            transform = ax[zi].transAxes, color='white'
            )
        
        ax[zi].set_xlabel(r'$T_{\rm 0 }{\rm [K]}$')
        ax[zi].legend(frameon=False, fontsize=30, handlelength=1, loc='upper left')
        ax[zi].tick_params(which='both',direction="in", width=1.5)
        ax[zi].tick_params(which='major',length=14, top=True, left=True, right=True)
        ax[zi].tick_params(which='minor',length=10, top=True, left=True, right=True)
        ax[zi].minorticks_on()
        ax[zi].set_xlabel(r'${\rm log}\Delta_{\rm \tau}$')
        
        
        if zi==0:
            ax[zi].set_ylabel(r'${\rm log(}{\rm T}_{\rm \tau} / {\rm [K]})$')
        else:
            ax[zi].set_yticklabels([])

###############################################################################

fig.savefig('td_obs'+'.pdf', format='pdf', dpi=90, bbox_inches = 'tight')

