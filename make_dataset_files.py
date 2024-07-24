import numpy as np
from read_skewers import read_skewers
import matplotlib.pyplot as plt

###############################################################################
# Simulated LOS files for dataset
###############################################################################

model_arr = [
    'planck1_20_1024',
    'planck1_20_1024_cold',    'planck1_20_1024_hot',
    'planck1_20_1024_zr525',   'planck1_20_1024_zr525_cold',   'planck1_20_1024_zr525_hot',
    'planck1_20_1024_zr675',   'planck1_20_1024_zr675_cold',   'planck1_20_1024_zr675_hot',
    'planck1_20_1024_zr750',   'planck1_20_1024_zr750_cold',   'planck1_20_1024_zr750_hot',
    'planck1_20_1024_g10',     'planck1_20_1024_g14',          'planck1_20_1024_g16',
    #'nyx_T682', 'nyx_T692'
]

# model_arr = [
#     'planck1_40_2048_RTzrfit',
#     'planck1_40_2048_RTzr53',       'planck1_40_2048_RTzr60',       'planck1_40_2048_RTzr67', 
#     'planck1_40_2048_RTzr53_homog', 'planck1_40_2048_RTzr60_homog', 'planck1_40_2048_RTzr67_homog',
#     ]

# model_arr = [
#         'planck1_40_2048',
#         'planck1_40_2048_cold',    'planck1_40_2048_hot',
#         'planck1_40_2048_zr750',   'planck1_40_2048_zr525',   'planck1_40_2048_zr675',
#     ]


#model_arr = ["nyx_zre6", "nyx_zre7", "nyx_zre7_hot", "nyx_zre7_cold", "nyx_zre8"]

# model_arr = ['planck1_160_1024', 'planck1_80_1024', 'planck1_40_1024', 'planck1_20_1024',
#              'planck1_160_2048', 'planck1_80_2048', 'planck1_40_2048']


dir_output = '/Users/nasir/Work/CNN/bh2igm/dataset_files/'

redshift_arr = [#4.0, 
                #4.4, 
                #5.0
                #4.2, 
                4.6, 
                #4.8
                ]
hubble = 0.676
observational = False

def make_files(model, mi, redshift):
    
    post ='_z'+"{:.2f}".format(redshift)+'.npy'
    
    dir_input = '/Volumes/Seagate/SherwoodData/'+model+'/'
    #dir_input = 'dataset_files/'
    
    if 'nyx' in model:
        filename = dir_input+model+post
        print(filename)
        with open(filename, 'rb') as f:
            opt = np.load(f)
            density = (np.load(f))
            temp = (np.load(f))
            densityw = (np.load(f))
            tempw = (np.load(f))
    else:
        density, temp, densityw, tempw, opt = read_skewers(dir_input, redshift, model)
        # density, temp, densityw, tempw, opt, nHI, vpec = read_skewers(dir_input, redshift, model)
    
    #sightlines = opt.shape[0]
    # filename = dir_output+'model_weights_z'+"{:.2f}".format(redshift)+'.npy'
    # with open(filename, 'rb') as f:
    #     model_weights = np.load(f)
    # model_weights = model_weights[mi*sightlines:(mi+1)*sightlines]
    # print('weights')
    # print(model_weights.shape, np.min(model_weights), np.max(model_weights))
    
    weights = np.full(opt.shape, 1.0)
    # weights = np.tile(model_weights[:, np.newaxis], (1,opt.shape[1]))
    # print('weights')
    # print(weights.shape, np.min(weights), np.max(weights))
    
    #Only take 80 percent and keep the first 20% slightlines as test dataset
    num_of_sightlines = density.shape[0]
    #slices_to_select = slice(0, num_of_sightlines)
    slices_to_select = slice(int(0.2*num_of_sightlines), num_of_sightlines)  #train/val dataset
    #slices_to_select = slice(0, int(0.2*num_of_sightlines)) #test dataset

    opt = opt[slices_to_select]
    density = density[slices_to_select]
    densityw = densityw[slices_to_select]
    temp = temp[slices_to_select]
    tempw = tempw[slices_to_select]
    weights = weights[slices_to_select]
 
    
    print('mean', np.mean(opt), np.mean(density), np.mean(densityw), 
          np.mean(temp), np.mean(tempw)
          )
    print('shapes', opt.shape, density.shape, temp.shape, 
          densityw.shape, tempw.shape
          )
    
    # Save multiple named arrays to the same file
    data_dict = {'opt': opt, 'density': density, 'temp': temp,
                  'densityw': densityw, 'tempw': tempw, 'weights': weights}
    
    # # Save multiple named arrays to the same file
    # data_dict = {'opt': opt, 'density': density, 'vpec': vpec, 'nHI': nHI, 'temp': temp, 
    #               'densityw': densityw, 'tempw': tempw, 'weights': weights}

    save_file = dir_output+'model_train_'+model+'_z'+"{:.2f}".format(redshift)+'.npy'
    print('saving ', save_file)
    with open(save_file, 'wb') as f:
          np.savez(f, **data_dict)


# for zi, redshift in enumerate(redshift_arr):
#     for mi, model in enumerate(model_arr):
#          make_files(model, mi, redshift)


###############################################################################
# Observational LOS files for dataset
###############################################################################

# Define pixel lists and redshifts
quasar_data = {
    
    #"J021043":[ {"pixels": [122, 280, 281, 564, 567, 619, 608],"redshift": 3.96}],
    
    #"J021043":[ {"pixels": [260, 261, 299, 300, 529, 528],"redshift": 4.0}],
    #"J025019":[ {"pixels": [615, 616, 971, 972, 87, 94, 99, 362, 363, 624, 656, 666, 667],"redshift": 4.0}],
    
    # "J021043":[ {"pixels": [425, 426],"redshift": 4.04}],
    # "J025019":[ {"pixels": [397, 950, 949],"redshift": 4.04}],
    # "J030722":[ {"pixels": [20, 21],"redshift": 4.04}],

    
    
    #"J021043":[ {"pixels": [344, 908, 909, 767, 768, 610, 611],"redshift": 4.16}],
    #"J025019":[ {"pixels": [29, 31, 27, 223, 243, 486, 484, 497, 656, 713, 714, 717, 718, 909, 907, 910, 347, 348, 908],"redshift": 4.16}],
    #"J033829":[ {"pixels": [147, 733],"redshift": 4.16}],
    
    # "J004054":[ {"pixels": [154, 155, 703, 704, 743, 744],"redshift": 4.2}],  
    # "J021043":[ {"pixels": [227, 228, 406, 513, 514, 496, 495, 364],"redshift": 4.2}],
    # "J025019":[ {"pixels": [532, 533, 534],"redshift": 4.2}],
    
    # "J004054":[ {"pixels": [108],"redshift": 4.24}],  
    # "J025019":[ {"pixels": [179, 180, 34, 35,181,179],"redshift": 4.24}],
    # "J030722":[ {"pixels": [696, 325, 326, 315, 316, 666, 667],"redshift": 4.24}],
    # "J033829":[ {"pixels": [553, 622],"redshift": 4.24}],
    # "J145147":[ {"pixels": [338],"redshift": 4.24}],
    
    
    #"J004054":[ {"pixels": [216, 214, 374, 373, 714, 260],"redshift": 4.35}],  
    #"J021043":[ {"pixels": [74, 373, 374, 488, 645, 650, 488, 882, 927, 921, 936],"redshift": 4.35}],  
    #"J033829":[ {"pixels": [300, 415, 1009, 1011, 689, 503, 502, 792],"redshift": 4.35}],  

    
    # "J004054":[ {"pixels": [508, 509, 855, 835, 836],"redshift": 4.4}],  
    # "J021043":[ {"pixels": [167, 399],"redshift": 4.4}],
    # "J030722": [{"pixels": [194, 776], "redshift": 4.4}],
    # "J025019":[ {"pixels": [298, 299, 880],"redshift": 4.4}],
    # "J033829":[ {"pixels": [28, 29, 80, 812, 813, 984, 918, 977, 978, 539, 546, 441, 448],"redshift": 4.4}],
    
        
    "J004054":[ {"pixels": [48],"redshift": 4.45}],  
    "J025019":[ {"pixels": [642, 643, 644, 36],"redshift": 4.45}],
    "J033829":[ {"pixels": [31],"redshift": 4.45}],
    
    
    # "J004054":[ {"pixels": [11, 13, 128, 275, 276, 374, 211],"redshift": 4.55}],  
    # "J021043":[ {"pixels": [600, 983],"redshift": 4.55}],  
    # "J025019":[ {"pixels": [144, 342],"redshift": 4.55}],
    # "J030722":[ {"pixels": [429, 430],"redshift": 4.55}],
    # "J033829":[ {"pixels": [682, 330, 331, 682, 841, 843, 7],"redshift": 4.55}],

    
    # "J030722": [{"pixels": [243, 395, 538, 632, 633, 1024, 1026], "redshift": 4.6}],
    # "J021043":[ {"pixels": [384, 552, 918, 917, 834, 934, 932, 872],"redshift": 4.6}],
    # "J004054":[ {"pixels": [15, 31, 465, 543, 544, 545, 715, 711],"redshift": 4.6}],  
    # "J025019":[ {"pixels": [85, 86, 1046, 731, 756, 755, 757, 769, 783, 773, 744, 721, 722],"redshift": 4.6}],
    # "J033829":[ {"pixels": [121, 472, 593, 613, 645, 872, 873, 1008, 1009],"redshift": 4.6}],
    
    #"J004054":[ {"pixels": [255, 256, 715],"redshift": 4.75}],  
    #"J033829":[ {"pixels": [627, 626],"redshift": 4.75}],
    
    #"J004054":[ {"pixels": [76, 77, 243, 437, 452, 552, 975, 976, 900, 901, 930, 644, 645, 690, 693, 792, 232, 233],"redshift": 4.8}],  
    #"J033829":[ {"pixels": [171, 181, 771, 772, 696, 693, 700, 581, 582, 67, 132, 181, 170],"redshift": 4.8}],

}


# Function to correct flux observations at specific pixels
def correct_flux_observations(flux_obs, qi, pixels_list):
    for pi in pixels_list:
        # Handle edge cases: make sure the indices are within bounds
        start_index = max(pi - 2, 0)
        end_index = min(pi + 2, flux_obs.shape[1])
        
        # Collect the neighborhood pixels
        neighborhood = np.concatenate((flux_obs[qi][start_index:pi], flux_obs[qi][pi+1:end_index]))
        
        # Update the flux at pixel pi with the median of its neighborhood
        flux_obs[qi][pi] = np.median(neighborhood)


# Function to find matching quasar data
def find_quasar_data(quasar, redshift):
    quasar_prefix = quasar[:7]
    if quasar_prefix in quasar_data:
        for entry in quasar_data[quasar_prefix]:
                        
            #print(f"Checking entry: {entry}")  # Debugging print
            if np.isclose(entry["redshift"], redshift, atol=1e-3):  # Adjust tolerance as needed
                return entry["pixels"]
    return None


redshift = 4.45
filename = dir_output+'obs'+'_z'+"{:.2f}".format(redshift)+'.npy'
print('reading obs file ', filename)
with open(filename, 'rb') as f:
    quasars = np.load(f)
    flux_obs = np.load(f)
    noise_obs = np.load(f)
    bad_obs = np.load(f)
    fwhm_obs = np.load(f)
    flux_bins = np.load(f)
    noise_model = np.load(f)


for qi, quasar in enumerate(quasars):
    
    mean_flux =  np.round(np.mean(flux_obs[qi][bad_obs[qi]==0]), 4)
    noise = np.round(np.mean(noise_obs[qi][bad_obs[qi]==0]), 2)
    fwhm = np.round(np.mean(fwhm_obs[qi]), 2)
    bins = flux_obs[qi].shape[0]
    

    # Apply corrections based on the quasar identifier and redshift
    pixels_list = find_quasar_data(quasar, redshift)
    
    if pixels_list:
        correct_flux_observations(flux_obs, qi, pixels_list)
        print(f"Applied corrections for {quasar[:7]} with redshift {redshift}")
    else:
        print(f"No matching data found for {quasar[:7]} with redshift {redshift}")

        
    # Save multiple named arrays to the same file
    data_dict = {'flux_obs': flux_obs[qi], 
                 'flux_level': flux_bins,
                  'noise_level': noise_model[qi], 
                  'mean_flux' : mean_flux,
                  'fwhm' : fwhm, 
                  'noise': noise,
                  'bins' : flux_obs[qi].shape[0],
                  'noise_obs': noise_obs[qi]}
    
    
    
    #output flux for predictions
    save_file = dir_output+quasar[:7]+'_z'+"{:.2f}".format(redshift)+'.npy'
    print('saving ', save_file)
    with open(save_file, 'wb') as f:
          np.savez(f, **data_dict)
          
          
    fig, ax = plt.subplots(1, 1, figsize=(28, 10))
    fig.subplots_adjust(wspace=0, hspace=0)
    
    axis = np.arange(flux_obs[qi].shape[0])
    
    ax.step(axis, flux_obs[qi], where='mid', color="black", alpha=1)
    ax.step(axis, noise_obs[qi], where='mid', color='orange', linestyle='--', 
                linewidth=2, alpha=.6)
    
    ax.set_xlim(0, axis[-1])
    ax.set_ylim(-0.04, 1.1)
    ax.set_ylabel(r'${\rm Flux}$')
    ax.text(0.01, .85, quasar[:7]+
               r', $\langle F \rangle =$'+"{:.2f}".format(mean_flux),
                    fontsize=32, transform = ax.transAxes, color="black")
    
    fig.savefig(quasar+'_z'+"{:.2f}".format(redshift)+'.pdf', 
          format='pdf', dpi=90, bbox_inches = 'tight')
    
    plt.show()

###############################################################################
# NYX skewers (3 skewers) trated as Observational LOS from Fred.
###############################################################################

'''
with open("dataset_files/model_test_nyx_test_A_z4.00.dat", 'rb') as f:
    flux_obs = np.loadtxt(f)
    
flux_obs = flux_obs[:,1]
plt.plot(flux_obs)

mean_flux =  np.round(np.mean(flux_obs), 4)
noise = np.round(0.02, 2)
fwhm = np.round(6.0, 2)
bins = 1024
redshift = 4.00

print('<F>=', mean_flux)

# get a flat noise model for the los
flux_bins = np.arange(-0.1, 1.1, 0.01)
noise_model = np.full(len(flux_bins), noise)

# Save multiple named arrays to the same file
data_dict = {'flux_obs': flux_obs, 
             'flux_level': flux_bins,
              'noise_level': noise_model, 
              'mean_flux' : mean_flux, 
              'fwhm' : fwhm, 
              'noise': noise,
              'bins' : bins}

#output flux for predictions
save_file = dir_output+'nyxa_z'+"{:.2f}".format(redshift)+'.npy'
print('saving ', save_file)
with open(save_file, 'wb') as f:
      np.savez(f, **data_dict)

'''
