import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from read_data_low_z_spikes import read_data_low_z_spikes_main
from adds import findNearest, noise_model, closest_argmin
from adds import mean_flux_z, mpch_to_dz

font = {'family' : 'serif', 'weight' : 'normal','size' : 30}
matplotlib.rc('font', **font)
dir_output = 'bh2igm/dataset_files/'
cl = ['black','red', 'blue', 'orange', 'purple', 'cyan', 'green']

###############################################################################
#read the selected sightlines from observational dataset .hdf5 file
###############################################################################

#remove the 6th index spectrum the fwhm is very different
#leaving out J160501-011220, index=6
selectedList = np.array([0, 1, 2, 3, 4, 5, 7, 8])
axisList, velaxisList, redshiftsList, fwhmList, snrList, fluxList, noise,\
                bad, names = read_data_low_z_spikes_main(selectedList)

###############################################################################
###############################################################################
'''
#plot the sightlines
fig, ax = plt.subplots(len(axisList), 1, figsize=(60, 6*len(axisList)))
fig.subplots_adjust(wspace=0.4, hspace=0.4)

for i in range(len(axisList)):
    quasar = names[i]
    
    #fluxList[i][(bad_array[i]==1)] = np.nan
    #noise_array[i][(bad_array[i]==1)] = np.nan
    #print(np.min(fwhmList[i]), np.max(fwhmList[i]))
    
    ax2 = ax[i].twiny()

    ax[i].plot(axisList[i], fluxList[i], color='black')
    #ax[i].plot(redshiftsList[i], noise[i], color='orange')
    #ax[i].fill_between(redshiftsList[i], bad[i], color='red', y2=0, alpha=.6)
    ax2.plot(redshiftsList[i], fluxList[i], color='black')
    #ax[i].plot(redshiftsList[i], fwhmList[i], color='black')
    ax[i].set_xlim(np.min(axisList[i]), np.max(axisList[i]))
    ax2.set_xlim(np.min(redshiftsList[i]), np.max(redshiftsList[i]))
    #ax[i].text(redshiftsList[i][10], 1, names[i])
    #ax[i].set_xlim(3.8, 3.9)
    #ax[i].set_ylim(-0.2, 1.2)
    ax[i].set_xlabel(r'$Mpc/h$') 
        
fig.savefig('carswell_los.pdf',format='pdf', dpi=90, bbox_inches = 'tight')
'''

###############################################################################
# make the noise and weigths for observationsl data to make mock datasets
###############################################################################

redshift_arr = np.array([4.0, 4.2, 4.4, 4.6, 4.8])

dz = 0.2
skewer_length = 20
hubble = 0.676
num_of_los = 5000

for i in range(len(redshift_arr)):
    redshift_arr[i] += mpch_to_dz(20.0, redshift_arr[i], hubble)

print(redshift_arr)

#the redshift segment of sightline to get chunck ont to which we need predictions
for zi, redshift in enumerate(redshift_arr):
    
    fluxArr = list()
    quasarArr = list()
    fwhmArr = list()
    noiseModelArr = list()
    noiseModelUpperArr = list()
    noiseModelLowerArr = list()
    badArr = list()
    initialized = False
    
    dzhalf = mpch_to_dz(skewer_length, redshift, hubble) * 0.5
    #bins for a given FWHM
    #bins = fwhm_to_bins( fwhm_arr[zi], skewer_length, redshift, hubble)
    print()
    print('z =', redshift)

    #get segments over each quasar we can make predictions
    for i in range(len(axisList)):
        quasar = names[i]
        
        #the medeium redshift and its pixel index
        index, zmed = findNearest(redshiftsList[i], redshift)
        
        #for initial assesssment of fwhm in the redshifts
        index_first, zfirst = findNearest(redshiftsList[i], redshift-dzhalf)
        index_last, zlast = findNearest(redshiftsList[i], redshift+dzhalf)
        bins = index_last - index_first
        dz_obs = zlast - zfirst
        
        ##fwhm_req = bins_to_fwhm(bins, skewer_length, redshift, hubble)
        axis = axisList[i]
        
        #if you been able to get a 99 percent of segment in 
        #redshift space fom sightline than save it
        #check if the LOS contains bad pixels for the majority 
        if dz_obs > 0.9*dzhalf*2.0 and \
        np.count_nonzero(bad[i][index_first:index_last]==1) < 0.5*bins: 
            print(quasar[:7], 'dz=', np.round(dz_obs, 3), 'dz bins = ',bins)
            
            if initialized==False:
                initialized = True
                quasarArr = quasar
                fluxArr = fluxList[i][index_first:index_last]
                badArr = bad[i][index_first:index_last]
                noiseArr = noise[i][index_first:index_last]
                fwhmArr = fwhmList[i][index_first:index_last]
                
            else:
                if len(fluxArr.shape)==2:
                    if bins!=fluxArr[0].shape:
                        index_last += (fluxArr[0].shape[0]-bins)
                        #print(fluxArr[0].shape[0],'not equal to', bins,'-->', index_last-index_first)
                else:
                    if bins!=len(fluxArr):
                        index_last += (len(fluxArr)-bins)
                        #print(len(fluxArr), bins,'not equal to','-->', index_last-index_first)
                        
                quasarArr = np.hstack((quasarArr, quasar))
                fluxArr = np.vstack((fluxArr, fluxList[i][index_first:index_last]))
                badArr = np.vstack((badArr, bad[i][index_first:index_last]))
                noiseArr = np.vstack((noiseArr, noise[i][index_first:index_last]))
                fwhmArr = np.vstack((fwhmArr, fwhmList[i][index_first:index_last]))

            #reset axis for plotting
            axis -= axis[-1]

    #removing singular and double isolated points and replaced with median noise within 3 pixel range
    if len(fluxArr)>0:
        #remove the one isolated bad pixel from flux
        for si in range(fluxArr.shape[0]):
            for pi in range(1, len(fluxArr[si])-1):
                if badArr[si][pi] == 1 and badArr[si][pi-1] == 0 and badArr[si][pi+1] == 0:
                    fluxArr[si][pi] = np.median([fluxArr[si][pi-1], fluxArr[si][pi+1]])
                    noiseArr[si][pi] = np.median([noiseArr[si][pi-1], noiseArr[si][pi+1]])
                    #set the pixel to non bad
                    badArr[si][pi] = 0
                    
        #remove the two isolated bad pixels from flux
        for si in range(fluxArr.shape[0]):
            for pi in range(1, len(fluxArr[si])-2):
                if badArr[si][pi] == 1 and badArr[si][pi-1] == 0 and badArr[si][pi+2] == 0:
                    fluxArr[si][pi:pi+1] = np.median([fluxArr[si][pi-1], fluxArr[si][pi+2]])
                    noiseArr[si][pi:pi+1] = np.median([noiseArr[si][pi-1], noiseArr[si][pi+2]])
                    #set the pixel to non bad
                    badArr[si][pi:pi+1] = 0

    
    initialized = False
        
    if len(fluxArr.shape)>1:
        #getting noise model for each LOS
        for i in range(fluxArr.shape[0]):        
            flux_bins, noise_med, _,_ = \
            noise_model(fluxArr[i], noiseArr[i], badArr[i])
                
            if initialized:
                noiseModelArr = np.vstack((noiseModelArr, noise_med))  
            else:
                initialized = True
                noiseModelArr = noise_med


    if len(fluxArr.shape)==1:
        flux_bins, noiseModelArr, _, _ = \
        noise_model(fluxArr.flatten(), noiseArr.flatten(), badArr.flatten())

    
    residual_noise =  np.full(fluxArr.shape, np.nan)
    realized_noise =  np.full(fluxArr.shape, np.nan)

    if len(fluxArr.shape)>1:  
        for i in range(fluxArr.shape[0]):
                realized_noise[i] = noiseModelArr[i][closest_argmin(fluxArr[i], flux_bins)]
                residual_noise[i] = noiseArr[i] - realized_noise[i]


    if len(fluxArr.shape)==1:
        realized_noise = noiseModelArr[closest_argmin(fluxArr.flatten(), flux_bins)]
        #print(noiseArr.shape, realized_noise.shape)

        residual_noise = noiseArr - realized_noise


    #threshold noise
    noiseArr[badArr==1]  = 0.0
    realized_noise[badArr==1] = 0.0
    residual_noise[badArr==1] = 0.0


    if len(fluxArr.shape)>1:
        for quasar in range(len(quasarArr)):
            mean_flux = np.mean(fluxArr[quasar][badArr[quasar]==0])
            fwhm = np.mean(fwhmArr[quasar])
            if len(fluxArr.shape)>1:
                bins = fluxArr.shape[1]
            else:
                 bins = len(fluxArr.flatten())
            snr = np.int32(np.mean(1.0/noiseModelArr[quasar][noiseModelArr[quasar]>0]))
    
            print(quasarArr[quasar][:7], '<F>=', mean_flux, mean_flux_z(redshift), 'fwhm=',
                      fwhm, 'S/N=',snr, 'bins=', bins)



    filename = dir_output+'obs'+'_z'+"{:.2f}".format(redshift)+'.npy'
    print('writing ', filename)
    with open(filename, 'wb') as f:
        np.save(f, quasarArr)
        np.save(f, fluxArr)
        np.save(f, noiseArr)
        np.save(f, badArr)
        np.save(f, fwhmArr)
        np.save(f, flux_bins)
        np.save(f, noiseModelArr)
        np.save(f, realized_noise)
        np.save(f, residual_noise)

######################################################################################################

    # fig, ax = plt.subplots(fluxArr.shape[0], 1, figsize=(24, 6*fluxArr.shape[0]))
    # fig.subplots_adjust(wspace=0.12, hspace=0.12)
    
    # # fig2, ax2 = plt.subplots(fluxArr.shape[0], 1, figsize=(10, 8*fluxArr.shape[0]))
    # # fig2.subplots_adjust(wspace=0.12, hspace=0.12)
    
    # axis = np.arange(noiseArr.shape[1])/np.float32(noiseArr.shape[1]) * skewer_length
    
    # for i in range(fluxArr.shape[0]):
        
    #     # ax2[i].scatter(fluxArr[i], noiseArr[i], color='black')
        
    #     # ax2[i].plot(flux_bins, noiseModelUpperArr[i], linestyle='--', color='orange')
    #     # ax2[i].plot(flux_bins, noiseModelLowerArr[i], linestyle='--', color='orange')
    #     # ax2[i].plot(flux_bins, noiseModelArr[i], color='orange')
        
    #     # ax2[i].scatter(fluxArr[i], realized_noise,  alpha=0.6, color='red')
    #     # #ax2[i].set_ylim(0., .12)
    #     # ax2[i].set_xlim(-0.2, 1.2)
    #     # ax2[i].set_xlabel(r'${\rm Flux}$')
    #     # ax2[i].set_ylabel(r'${\rm Noise}$')
    #     # #ax2.set_ylabel(r'${\rm S/N}$')
        
    #     #ax[i].step(axis, fluxArr[i], color='black')
    #     ax[i].plot(axis, realized_noise[i], color='red', linestyle='--')        
    #     ax[i].step(axis, noiseArr[i], color='orange', linestyle='--')
    #     ax[i].step(axis, residual_noise[i], color='blue', linestyle='--')

    #     #ax[i].step(axis, badArr[i], color='red', linestyle='--', alpha=1)
    #     ax[i].set_xlim(axis[0], axis[-1])
    #     ax[i].set_xlabel('')
    #     ax[i].set_ylabel(r'${\rm Flux}$')
    #     noiseSeg = noiseArr[i]
    #     ax[i].text( 0.02, 0.9, quasarArr[i]+r'$\,\,{\rm S/N}=$+'+str(np.round(1/np.mean(noiseSeg[noiseSeg>0])))+\
    #                 ', FWHM='+str(np.round(np.mean(fwhmArr[i]), 3))+' z='+str(redshift)+\
    #                     ', <F>='+str(np.round(np.mean(fluxArr[i][badArr[i]==0]), 3))+\
    #                         ', bins='+str(fluxArr[i].shape[0]),\
    #                             fontsize=18, transform = ax[i].transAxes)

        
    # fig.savefig('obs_los_z'+"{:.2f}".format(redshift)+'.pdf',format='pdf', dpi=90, bbox_inches = 'tight')
       
