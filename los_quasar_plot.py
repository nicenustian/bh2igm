import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import FormatStrFormatter
from adds import t0_gamma, findNearest
#from adds import t0_gamma_from_bs_corr
from adds import read_predictions, t0_gamma_from_bs, get_t0_gamma_values_str
#from adds import t0_gamma_to_sim_model
#from scipy.linalg import eigh, cholesky
rng = np.random.default_rng()

font = {'family' : 'serif', 'weight' : 'normal','size' : 34}
matplotlib.rc('font', **font)

# cl = ['red','blue','orange','purple','olive','tan','green','magenta']
cl = ['black','black','black','black','black','black','black','black']
ls = ['--','--','--', '--', '--5']
lw = 2

skewer_length = 20

#for noise generation
seed = 12345
np.random.seed(seed)
hubble = 0.678

quasar = "J004054"
dir_output = 'bh2igm/'+"ml_outputs_J004054_mflux0.1213_fwhm5.88_bins1075_noise0.02_z4.80"+"/"

dir_dataset = 'bh2igm/dataset_files/'
realizations = 1000
trim = 16

###############################################################################

los = 0
redshifts     = [4.8]
obs_redshifts = [4.8]

for zi, redshift in enumerate(redshifts):
        
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
                
        snr = 1/np.mean(noise_obs)
        
        fig, ax = plt.subplots(3, 1, figsize=(28, 4*3))
        fig.subplots_adjust(wspace=0, hspace=0)
        
        #qs_color = cl[quasar_list.index(quasar[:7])]
        pixels = len(flux)
                
        flux_los, densityw_mean, densityw_upper_1sigma , densityw_lower_1sigma, \
        tempw_mean, tempw_upper_1sigma , tempw_lower_1sigma = \
            read_predictions(dir_dataset, dir_output, 
                             quasar, redshift, False)
            
        
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
        
        sim_model = 'planck1_20_1024'
            
        filename = dir_output+"/"+'corr_'+sim_model+'.npy'
        print(filename)
        corr = np.load(filename)
        pixels = densityw_mean.shape[1]
     
        concat_real_orig = np.random.multivariate_normal(
                np.zeros(pixels*2), corr, realizations)
        
        concat_real = mean_concat[los] + concat_real_orig * std_concat[los]
            
        t0 = np.full(realizations, np.nan)
        gamma = np.full(realizations, np.nan)
            
        for ri in range(realizations):
            t0[ri], gamma[ri] = t0_gamma(concat_real[ri,  :pixels][mask], 
                             concat_real[ri, pixels:][mask])
            
        crl, crm, cru = np.quantile(concat_real, [0.16, 0.5, 0.84], axis=0)
        
        t0m, gammam, tempstr, gammastr = get_t0_gamma_values_str(t0, gamma)                
        print('with corr = ', tempstr, gammastr)        
        print('densityw_mean', densityw_mean.shape)

        
        dx = skewer_length/len(flux)/hubble
        axis = np.arange(len(flux)) * (skewer_length/len(flux))
        axis_orig = np.arange(len(densityw_mean[los])) * (skewer_length/len(densityw_mean[los]))
    
    
        qs_color = 'black' #cl[quasar_list.index(quasar)]
        ax[0].step(axis, flux, where='mid', color=qs_color, alpha=1)
        ax[0].step(axis, noise_obs, where='mid', color='orange', linestyle='--', 
                    linewidth=lw, alpha=.6)
        
        ax[0].set_xlim(0, skewer_length)
        ax[0].set_ylim(-0.04, 1.1)
        ax[0].set_ylabel(r'${\rm Flux}$')
        ax[0].text(0.01, .85, quasar+r'$, \,{\rm S/N}=$'+str(np.int32(snr))+
                   r', $\langle F \rangle =$'+"{:.2f}".format(mean_flux),
                        fontsize=32, transform = ax[0].transAxes, color=qs_color)
        
        
        ax[1].set_xlim(0, skewer_length)
        ax[1].axhline(0, linestyle='--', color="orange", alpha=0.6 )
        ax[1].step(axis_orig, densityw_mean[los], where='mid', color=qs_color, alpha=1)
        ax[1].fill_between(axis_orig, cru[:pixels], y2=crl[:pixels], color=qs_color, 
                            alpha=.2)
        #ax[1].set_ylim(-1.2, 1.2)
        ax[1].set_ylabel(r'${\rm log}\Delta_{\rm \tau}$')
        #ax[2].set_ylim(3.2, 4.8)
        ax[2].set_ylabel(r'${\rm log(}{\rm T}_{\rm \tau} / {\rm [K]})$')

        ax[2].set_xlim(0, skewer_length)
        ax[2].step(axis_orig, crm[pixels:], where='mid', linestyle='-', alpha=1, color=qs_color)
        ax[2].fill_between(axis_orig, cru[pixels:], y2=crl[pixels:], color=qs_color, 
                            alpha=.2)
        ax[2].text(0.01, 0.8, r'$T_{\rm 0}$=' + tempstr + '  ' + r'$\gamma=$'+gammastr,
                        fontsize=32, transform = ax[2].transAxes, color=qs_color)
    
    
        for pi in range(3):
            ax[pi].tick_params(which='both',direction="in", width=1.5)
            ax[pi].tick_params(which='major',length=14, top=True, left=True, right=True)
            ax[pi].tick_params(which='minor',length=10, top=True, left=True, right=True)
            ax[pi].minorticks_on()
           
        ax[0].set_xlabel('')
        ax[1].set_xlabel('')
        ax[2].set_xlabel(r'${\rm Mpc/{\rm h}}$')
        ax[0].set_xticklabels([])
        ax[1].set_xticklabels([])
        ax[1].legend(frameon=False, fontsize=32, 
                      bbox_to_anchor=(0.48, 1.1),
                      handlelength=1, loc='upper center')
        
        
        #plt.show()
        
        fig.savefig(quasar+'_snr'+str(np.int32(snr))+'_z'+"{:.2f}".format(
            obs_redshifts[zi])+'.pdf', 
                    format='pdf', dpi=90, bbox_inches = 'tight')
        plt.close()
        
