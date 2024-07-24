import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from adds import t0_gamma_from_bs_corr, get_contours, findNearest
from adds import read_predictions, t0_gamma_from_bs, get_t0_gamma_values_str
from adds import t0_gamma_to_sim_model, joint_conf, t0_gamma
import os
import os.path
#from matplotlib.ticker import FormatStrFormatter
from cov_models import get_cov

font = {'family' : 'serif', 'weight' : 'normal','size' : 34}
matplotlib.rc('font', **font)

lw = 2
skewer_length = 20
hubble = 0.678
realizations = 1000

trim = 16
los = 0

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

#redshifts = [4.04, 4.24, 4.45, 4.65]

quasar_list = ['J021043', 'J025019', 'J145147','J120523', 
               'J004054', 'J025019', 'J030722', 'J033829']
cl = ['red', 'blue', 'orange','purple', 'olive', 'tan', 'green','magenta']

###############################################################################
#load UVB models
###############################################################################

fg20 = np.loadtxt('UVB_Models/Thermal_Parameter_FG20.txt', dtype=np.float32)
hm12 = np.loadtxt('UVB_Models/Thermal_Parameter_HM12.txt', dtype=np.float32)
ks19 = np.loadtxt('UVB_Models/Thermal_Parameter_KS19.txt', dtype=np.float32)
oh17 = np.loadtxt('UVB_Models/Thermal_Parameter_OH17.txt', dtype=np.float32)
p19 =  np.loadtxt('UVB_Models/Thermal_Parameter_P19.txt', dtype=np.float32)

b19 = np.loadtxt('Observation/boera_2019.txt', dtype=np.float32)
w18 = np.loadtxt('Observation/walther_2018.txt', dtype=np.float32, 
                 usecols = (0,2,3,4,5,6,7))
g21 = np.loadtxt('Observation/gaikwad_2021.txt', dtype=np.float32, 
                 usecols = (0,2,3,4,5,6,7))
g20 = np.loadtxt('Observation/gaikwad_2020.txt', dtype=np.float32, 
                 usecols = (0,2,3,4,5,6,7))

###############################################################################

#fig, ax = plt.subplots(1, 2, figsize=(16, 8))
fig, ax = plt.subplots(1, len(redshifts), figsize=(10*len(redshifts), 8))
fig.subplots_adjust(wspace=0.0, hspace=.0)

fig2, ax2 = plt.subplots(1, 2, figsize=(20, 8))
fig2.subplots_adjust(wspace=0.3, hspace=.0)

combined_densityw_mean = [ [] for _ in range(len(redshifts)) ]
combined_tempw_mean = [ [] for _ in range(len(redshifts)) ]
combined_densityw_std = [ [] for _ in range(len(redshifts)) ]
combined_tempw_std = [ [] for _ in range(len(redshifts)) ]
combined_corr = [ [] for _ in range(len(redshifts)) ]
combined_t0 = [ [] for _ in range(len(redshifts)) ]
combined_gamma = [ [] for _ in range(len(redshifts)) ]
combined_mask = [ [] for _ in range(len(redshifts)) ]

initialized = [False] * len(redshifts)


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
    
    print()
    print()
    
    if snr>=0:# and quasar!='J145147': #and redshift==4.0:
    
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
        
        #sim_model = 'planck1_20_1024'
            
        filename = dir_output+"/"+file_folder+"/"+'corr_'+sim_model+'.npy'
        
        if os.path.isfile(filename):
            corr = np.load(filename)
            #print('file for correlation..', corr)
        else:
            corr = get_cov(dir_dataset+"/", dir_output+"/"+file_folder+"/", sim_model, redshift)

        
        
        pixels = densityw_mean.shape[1]
     
        concat_real_orig = np.random.multivariate_normal(
                np.zeros(pixels*2), corr, realizations)
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
        #print('densityw_mean', densityw_mean.shape)
                                                                
    
        if initialized[zi] == False:
            initialized[zi] = True
                    
            combined_densityw_mean[zi] = densityw_mean
            combined_tempw_mean[zi] = tempw_mean
            combined_densityw_std[zi] = densityw_std
            combined_tempw_std[zi] = tempw_std
    
            combined_corr[zi] = corr
            combined_mask[zi] = mask
            #each LOS is an independent measurement, find joint PDF and than KDE
            combined_t0[zi] = t0
            combined_gamma[zi] = gamma
            
        else:
                        
            combined_densityw_mean[zi] = np.vstack((combined_densityw_mean[zi], densityw_mean))
            combined_tempw_mean[zi] = np.vstack((combined_tempw_mean[zi], tempw_mean))
            combined_densityw_std[zi] = np.vstack((combined_densityw_std[zi], densityw_std))
            combined_tempw_std[zi] = np.vstack((combined_tempw_std[zi], tempw_std))
    
            combined_corr[zi] = np.dstack((combined_corr[zi], corr))
            #each LOS is an independent measurement, find joint PDF and than KDE
            combined_t0[zi] = np.vstack((combined_t0[zi], t0))
            combined_gamma[zi] = np.vstack((combined_gamma[zi], gamma))
            combined_mask[zi] = np.vstack((combined_mask[zi], mask))
            
        
        #######################################################################
     
        X,Y,Z,conf = get_contours(t0, gamma)       
        ##ax[zi].scatter(10**t0[:1000], gamma[:1000], marker='.', alpha=0.04, c=qs_color)
        ax[zi].contour(X,Y,Z, conf, colors=qs_color, alpha=0.6, linestyles='solid')
        #ax.yaxis.get_major_locator().set_params(integer=True)
        ax[zi].set_xlim(2999, 14900)
        ax[zi].set_ylim(.89, 2.)
        
        ax[zi].text(0.65, 0.04, r'$z=$'+"{:.2f}".format(redshift), 
                        fontsize=28, transform = ax[zi].transAxes, 
                        color='black')
    
        # ax[zi].text(0.1, 1.08+qso_index*0.06, r'${\rm S/N=}$'+str(snr)+ r', $T_{\rm 0}$=' + 
        #              tempstr + r'$, \gamma$=' + gammastr,
        #                fontsize=20, transform = ax[zi].transAxes, color=qs_color)
    
        ax[zi].text(0.04,0.04+qso_index*0.06, 
                      quasar+r', $T_{\rm 0}$=' +\
                          str(int(np.mean(10**t0)))+r'${\rm K}$' + ', ' + r'$\gamma=$'\
                              +str(np.round(np.mean(gamma), 2))\
                          +', '+r'${\rm S/N}=$'+str(int(snr))+\
                              ', '+r'${\rm \langle F \rangle}=$'+str(np.round(np.mean(flux[mask]), 2)),
                      fontsize=20, transform = ax[zi].transAxes, color=qs_color)
    
        ax[zi].set_xlabel(r'$T_{\rm 0 }{\rm [K]}$')
        ax[zi].legend(frameon=False, fontsize=30, handlelength=1, loc='upper left')
        ax[zi].tick_params(which='both',direction="in", width=1.5)
        ax[zi].tick_params(which='major',length=14, top=True, left=True, right=True)
        ax[zi].tick_params(which='minor',length=10, top=True, left=True, right=True)
        ax[zi].minorticks_on()
    
        if zi==0:
            ax[zi].set_ylabel(r'$\gamma$')
        else:
            ax[zi].set_yticklabels([])
     
        ax2[0].errorbar(redshift, t0m[0], yerr=
                          np.array([np.abs(t0m[2]),t0m[1]]).reshape(2,1), 
                          fmt='o', color=qs_color,  ecolor=qs_color, alpha=0.4)
    
        ax2[0].set_xlabel(r'$z$')
        ax2[0].set_ylabel(r'$T_{\rm 0}{\rm [K]}$')
        ax2[0].set_xlim(3.4, 5.8)
        
        ax2[1].errorbar(redshift, gammam[0], 
                          yerr=np.array([np.abs(gammam[2]), np.abs(gammam[1])]).reshape(2,1), 
                          fmt='o', color=qs_color, ecolor=qs_color, alpha=0.4)
        
        
        ax2[1].set_xlabel(r'$z$')
        ax2[1].set_ylabel(r'$\gamma$')
        ax2[1].set_xlim(3.4, 5.8)
    

for zi, redshift in enumerate(redshifts):
    
    if len(combined_densityw_mean[zi])>0:
        
        t0, gamma = t0_gamma_from_bs_corr(
            combined_densityw_mean[zi], combined_densityw_std[zi],
            combined_tempw_mean[zi], combined_tempw_std[zi], 
            combined_corr[zi], combined_mask[zi], 10000)
            
    
    #X, Y, Z, conf, t0, gamma = joint_conf(combined_t0[zi], combined_gamma[zi])    
    X,Y,Z,conf = get_contours(t0, gamma)
    t0m, gammam, tempstr, gammastr = get_t0_gamma_values_str(t0, gamma, False)
    
    
    t0_error=np.abs(np.array([t0m[2], t0m[1]]).reshape((2,1)))
    gamma_error=np.abs(np.array([gammam[2], gammam[1]]).reshape((2,1)))
    
    #ax[zi].scatter(10**t0[:1000], gamma[:1000], marker='.', alpha=0.04, c='black')
    ax[zi].contour(X,Y,Z, conf, colors='black', alpha=1, linestyles='solid')    
    ax[zi].text(0.04, 0.9, r'$T_{\rm 0}$=' + tempstr+r'$\gamma=$'+gammastr,
                  fontsize=28, transform = ax[zi].transAxes, color='black')  
        

    if zi==0:
        bar_label='This work'
    else:
        bar_label=''
    
    # ax2[0].errorbar(redshift-0.01, t0m[0], yerr=t0_error, 
    #                                 fmt='o', ecolor='black', color='black', alpha=1,
    #                                 markersize='14', elinewidth=4,  label=bar_label)
    
    # ax2[1].errorbar(redshift-0.01, gammam[0], yerr=gamma_error, 
    #                             fmt='o', ecolor='black', color='black', alpha=1, 
    #                             label=bar_label, markersize='14', elinewidth=4)



# #renormlize the heating rates from Gaikwad2021
# # fg20[:, 1] /= 0.7 
# # p19[:, 1] /= 0.9
# # oh17[:, 1] /= 0.8

ax2[0].plot(fg20[:,0], fg20[:,1], color='blue',
            linestyle='-', label=r'${\rm FG20 x 0.7}$', linewidth=lw)
ax2[0].plot(p19[:,0],  p19[:,1],  color='red',
            linestyle='--', label=r'${\rm P19 x 0.9}$', linewidth=lw)
ax2[0].plot(hm12[:,0], hm12[:,1], color='purple',
            linestyle=':', label=r'${\rm HM12}$', linewidth=lw)
ax2[0].plot(ks19[:,0], ks19[:,1], color='black',
            linestyle='-.', label=r'${\rm KS19}$', linewidth=lw)
ax2[0].plot(oh17[:,0], oh17[:,1], color='green',
            linestyle='-', label=r'${\rm O17 x 0.8}$', linewidth=lw)


ax2[0].errorbar(b19[:,0]-0.01, b19[:,2], yerr=[b19[:,3],b19[:,4]], fmt='^',
                color='blue',  ecolor='blue', alpha=1, label='Boera2019', 
                markersize='10', elinewidth=2)
ax2[0].errorbar(w18[:,0]+0.01, w18[:,1], yerr=[w18[:,2],w18[:,3]], fmt='*', 
                color='red',  ecolor='red', alpha=1,label='Walther2018', 
                markersize='10', elinewidth=2)
ax2[0].errorbar(g21[:,0]-0.01, g21[:,1], yerr=[g21[:,2],g21[:,3]], fmt='o',
                color='orange',  ecolor='orange', label='Gaikwad2021', 
                markersize='10', elinewidth=2, alpha=1)
ax2[0].errorbar(g20[:,0]-0.01, g20[:,1], yerr=[g20[:,2],g20[:,3]], fmt='<', 
                color='purple',  ecolor='purple', label='Gaikwad2020', 
                markersize='10', elinewidth=2, alpha=1)

ax2[0].set_ylim(5000, 14000)
ax2[0].legend(loc='upper center', fontsize=28, handlelength=1,
              bbox_to_anchor=(1.1, 1.3), ncol=5, fancybox=True)

ax2[1].plot(fg20[:,0], fg20[:,2], color='blue',  linestyle='-',  linewidth=lw)
ax2[1].plot( p19[:,0],  p19[:,2], color='red',   linestyle='--', linewidth=lw)
ax2[1].plot(hm12[:,0], hm12[:,2], color='purple',linestyle=':',  linewidth=lw)
ax2[1].plot(ks19[:,0], ks19[:,2], color='black', linestyle='-.', linewidth=lw)
ax2[1].plot(oh17[:,0], oh17[:,2], color='green', linestyle='-',  linewidth=lw)


ax2[1].errorbar(b19[:,0]-0.01, b19[:,5], yerr=[b19[:,6],b19[:,7]], fmt='^', 
                color='blue',  ecolor='blue', alpha=1, markersize='10', 
                elinewidth=2)
ax2[1].errorbar(w18[:,0]+0.01, w18[:,4], yerr=[w18[:,5],w18[:,6]], fmt='*', 
                color='red',  ecolor='red', alpha=1, markersize='10', 
                elinewidth=2)
ax2[1].errorbar(g21[:,0]-0.01, g21[:,4], yerr=[g21[:,5],g21[:,6]], fmt='o', 
                color='orange',  ecolor='orange', alpha=1, markersize='10', 
                elinewidth=2)
ax2[1].errorbar(g20[:,0]+0.01, g20[:,4], yerr=[g20[:,5],g20[:,6]], fmt='<', 
                color='purple',  ecolor='purple', alpha=1, markersize='10', 
                elinewidth=2)

ax2[1].set_ylim(.99, 2.)

###############################################################################

for pi in range(2):
    ax2[pi].tick_params(which='both',direction="in", width=1.5)
    ax2[pi].tick_params(which='major',length=14, top=True, left=True, right=True)
    ax2[pi].tick_params(which='minor',length=10, top=True, left=True, right=True)
    ax2[pi].minorticks_on()
    
###############################################################################

fig.savefig('thermal_params_obs_joint'+'.pdf',
              format='pdf', dpi=90, bbox_inches = 'tight')

fig2.savefig('evolution_obs_joint'+'.pdf', 
              format='pdf', dpi=90, bbox_inches = 'tight')
