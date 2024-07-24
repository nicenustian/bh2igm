import numpy as np
import scipy.integrate as intg
from numba import njit
#import tensorflow as tf
import scipy.stats as st
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
#from scipy.linalg import cholesky
#from numpy.random import standard_normal
from poly_fit_numba import fit_poly, eval_polynomial
from scipy.ndimage.filters import gaussian_filter


h= 0.67
thompson    = 6.28e-18 #cm2 (Osterbrock & Ferland 2006)
MPC2CM      = 3.086e24
OMEGAM      = 0.305147
OMEGAB      = 0.0482266
HUBBLE      = 0.676
KPC2CM      = 3.086e21
fracb       = OMEGAB/OMEGAM
speedOfLight = 2.99792e5


#scale for calculating MFP
scale  = 0.032 #Mpc/h
numlos = 10000
bins   = 1024
Ngas   = 1024
dxh    = 1.024/Ngas

sigma_HI = 2.51e-18
chi = 0.08
T = 1e4


@njit
def sigma_cov(y_true, y_pred, sigma_pred):
    
    sightlines = y_true.shape[0]
    Npixels = y_true.shape[1]
    
    sigma_count = np.zeros(sightlines)
    
    for los in range(sightlines):
         sigma_count[los] = np.count_nonzero((y_true[los]>=(y_pred[los] - sigma_pred[los])) & 
                                  (y_true[los]<=(y_pred[los] + sigma_pred[los])))
    return sigma_count.astype(np.float32)/Npixels


def get_t0_gamma_values_str(t0, gamma, quantile_flag=False):
    
    if quantile_flag==False:
        t0l, t0m, t0u = np.quantile(t0, [0.16, 0.5, 0.84])
        gammal, gammam, gammau = np.quantile(gamma, [0.16, 0.5, 0.84])
        
        gammav, gammastr = get_dens_values_str(gamma)
        t0v, tempstr = get_temp_values_str(t0)
        
    else:
        gammav, gammastr = get_dens_values_str(gamma, True)
        t0v, tempstr = get_temp_values_str(t0, True)


    return t0v, gammav, tempstr, gammastr



def get_temp_values_str(t0, quantile_flag=False):
    
    if not quantile_flag:
        t0l, t0m, t0u = np.quantile(t0, [0.16, 0.5, 0.84])
    else:
        t0l, t0m, t0u = t0
     
    t0u = (10**t0u-10**t0m)
    t0l = (10**t0l-10**t0m)
    t0m = (10**t0m)
    
    kelvin_str = r'{\rm K}'
    
    #check if entity positvie or negatvie
    if t0u>=0:
        t0us = '+'+str(np.int32(t0u)) + kelvin_str 
    else:
        t0us = str(np.int32(t0u)) + kelvin_str 

    if t0l>=0:
        t0ls = '+'+str(np.int32(t0l)) + kelvin_str 
    else:
        t0ls = str(np.int32(t0l)) + kelvin_str 

    
    t0ms = str(np.int32(t0m)) + kelvin_str
            
    tempstr = ("${{{}}}^{{{}}}_{{{}}}$").format(t0ms, t0us, t0ls)

    return [t0m, t0u, t0l], tempstr



def get_dens_values_str(t0, quantile_flag=False):
    
    if quantile_flag==False:
        t0l, t0m, t0u = np.quantile(t0, [0.16, 0.5, 0.84])
    else:
        print('values to unpack=', t0)
        t0l, t0m, t0u = t0
        
    t0u = (t0u-t0m)
    t0l = (t0l-t0m)
    t0m = (t0m)
    
    #check if entity positvie or negative
    if t0u>=0:
        t0us = '+'+str(np.round((t0u), 2))
    else:
        t0us = str(np.round((t0u), 2)) 

    if t0l>=0:
        t0ls = '+'+str(np.round((t0l), 2))
    else:
        t0ls = str(np.round((t0l), 2))

    t0ms = str(np.round((t0m), 2))
    denstr =("${{{}}}^{{{}}}_{{{}}}$").format(t0ms, t0us, t0ls)

    return [t0m, t0u, t0l], denstr


def read_predictions(dir_dataset, dir_output, model, redshift, not_observational=True):
    
    if not_observational==True:
 
        if 'nyx_' in model or '_40_' in model:
            post_fix = 'model_test_'
            
        else:
            post_fix = 'model_test_'
            
        
        save_file = dir_dataset+post_fix+model+'_z'+"{:.2f}".format(redshift)+'.npy'
        print(save_file)
        data = np.load(save_file,'rb')
                
    if not_observational==True:
        filename_densityw = dir_output+'predict_flux_densityw_'+post_fix+model+ '_z'+"{:.2f}".format(redshift)+'.npy'
        filename_tempw = dir_output+'predict_flux_tempw_'+post_fix+model+ '_z'+"{:.2f}".format(redshift)+'.npy'

    else:
        filename_densityw = dir_output+'predict_'+model+'_flux_densityw'+'.npy'
        filename_tempw = dir_output+'predict_'+model+'_flux_tempw'+'.npy'


    print(filename_densityw)
    with open(filename_densityw, 'rb') as f:
        flux = np.load(f)
        if not_observational:
            densityw = np.load(f)
        densityw_mean = np.load(f)
        densityw_upper_1sigma = np.load(f)
        densityw_lower_1sigma = np.load(f)
    
    
    print(filename_tempw)
    with open(filename_tempw, 'rb') as f:
        flux = np.load(f)
        if not_observational:
            tempw = np.load(f)
        tempw_mean = np.load(f)
        tempw_upper_1sigma = np.load(f)
        tempw_lower_1sigma = np.load(f)
        
    if not_observational==True:
        return flux, data["density"], data["temp"], densityw, tempw, densityw_mean, densityw_upper_1sigma , \
    densityw_lower_1sigma, tempw_mean, tempw_upper_1sigma , tempw_lower_1sigma 
    
    else:
        return flux, densityw_mean, densityw_upper_1sigma , densityw_lower_1sigma, \
        tempw_mean, tempw_upper_1sigma , tempw_lower_1sigma
        

def get_contours(t0, gamma):
        
    xmin = np.min(10**t0) - 1000
    xmax = np.max(10**t0) + 1000
    
    ymin = np.min(gamma) - 0.1
    ymax = np.max(gamma) + 0.1
            
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([10**t0, gamma])
            
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    conf = find_conf_int(Z)
    
    return X,Y,Z, conf



#@njit
def joint_pdf(t0, gamma):
    
    xmin = np.min(t0) - 0.1
    xmax = np.max(t0) + 0.1
    
    ymin = np.min(gamma) - 0.1
    ymax = np.max(gamma) + 0.1
    
    binx = (xmax-xmin)/40
    biny = (ymax-ymin)/40
    
    xedges = np.arange(xmin, xmax, binx)
    yedges = np.arange(ymin, ymax, biny)
    
    x_center = xedges[1:] + 0.5*binx
    y_center = yedges[1:] + 0.5*biny

    joint_hist = np.zeros((t0.shape[0], len(xedges)-1, len(yedges)-1))
    for los in range(t0.shape[0]):
        
        values = np.vstack([t0[los], gamma[los]])
        kde = stats.gaussian_kde(values)
        joint_hist[los] = kde.evaluate([x_center, y_center])


    joint_hist = np.prod(joint_hist, axis=0)
    # confidence intervals for each X and Y random varaible
    x_counts = np.sum(joint_hist, axis=1)
    y_counts = np.sum(joint_hist, axis=0)
    
    xsum = np.float32(np.sum(x_counts))
    ysum = np.float32(np.sum(y_counts))
    
    if xsum!=0 and ysum!=0:
        x_cdf = np.cumsum(x_counts)/xsum
        y_cdf = np.cumsum(y_counts)/ysum

        x_quant = np.interp([0.5], x_cdf, x_center)
        y_quant = np.interp([0.5], y_cdf, y_center)
        
    else:
        x_quant = np.nan
        y_quant = np.nan


    return x_quant, y_quant



def joint_conf(t0, gamma):

    print('the t0 gamma shapes', t0.shape, gamma.shape)

    xmin = np.min(10**t0) - 1000
    xmax = np.max(10**t0) + 1000

    ymin = np.min(gamma) - 0.1
    ymax = np.max(gamma) + 0.1

    binx = (xmax-xmin)/200
    biny = (ymax-ymin)/200

    xedges = np.arange(xmin, xmax, binx)
    yedges = np.arange(ymin, ymax, biny)

    x_center = xedges[1:] + 0.5*binx
    y_center = yedges[1:] + 0.5*biny

    if len(t0.shape)>=2:
        joint_hist = np.zeros((t0.shape[0], len(xedges)-1, len(yedges)-1))
        for los in range(t0.shape[0]):
            joint_hist[los], _, _ = np.histogram2d(
                10**t0[los], gamma[los], bins=(xedges, yedges))
            
        joint_hist = np.prod(joint_hist, axis=0)
    else:
        joint_hist = np.zeros((len(xedges)-1, len(yedges)-1))
        joint_hist, _, _ = np.histogram2d(
            10**t0, gamma, bins=(xedges, yedges))

    
    # confidence intervals for each X and Y random varaible
    x_counts = np.sum(joint_hist, axis=1)
    y_counts = np.sum(joint_hist, axis=0)

    sum_xcounts = np.float32(np.sum(x_counts))
    sum_ycounts = np.float32(np.sum(y_counts))

    x_cdf = np.cumsum(x_counts)/sum_xcounts
    y_cdf = np.cumsum(y_counts)/sum_ycounts

    # to obtain 1sigma confidence interval for t0 and gamma
    x_quant = np.interp([.16, .5, .84], x_cdf, x_center)
    y_quant = np.interp([.16, .5, .84], y_cdf, y_center)

    conf = find_conf_int(joint_hist)
    joint_hist = gaussian_filter(joint_hist, 2)

    return x_center, y_center, joint_hist.T, conf, np.log10(x_quant), y_quant



#only for 1d LOS
##@njit
def t0_gamma_from_bs_corr(
        densityw_mean, densityw_std, 
        tempw_mean, tempw_std, 
        corr, mask=None, realizations=100
        ):
    
    sightlines = densityw_mean.shape[0]
    pixels = densityw_mean.shape[1]
    concat_real = np.full((sightlines, realizations, pixels*2), np.nan)

    t0 = np.full(realizations, np.nan)
    gamma = np.full(realizations, np.nan)
    mean_concat = np.hstack((densityw_mean, tempw_mean))
    std_concat = np.hstack((densityw_std, tempw_std))

    ##L = cholesky(corr)
    
    #joint TD plane estimate of mutiple obs LOS
    for los in range(sightlines):
        if sightlines>1:
            corr_matrix = corr[:,:,los]
        else:
            corr_matrix = corr

        #concat_real[los] =  np.random.multivariate_normal(
        #            mean_concat[los], corr_matrix, realizations)
        
        concat_real[los] = mean_concat[los] + np.random.multivariate_normal(
                 np.zeros(mean_concat[los].shape), corr_matrix, realizations)*std_concat[los]

    if mask is not None:
        mask = mask.reshape(sightlines, pixels)
    
    for ri in range(realizations):    
        if mask is not None:
            t0[ri], gamma[ri] = \
                t0_gamma(concat_real[:, ri, :pixels][mask], concat_real[:, ri, pixels:][mask])
            
        else:          
            t0[ri], gamma[ri] = \
                t0_gamma(concat_real[:, ri, :pixels], concat_real[:, ri, pixels:])
            
            
    return t0, gamma


@njit
def t0_gamma_from_bs(densityw_mean, densityw_std, tempw_mean, tempw_std, realizations=100):
            
    t0 = np.full(realizations, np.nan)
    gamma = np.full(realizations, np.nan)
    
    for ri in range(realizations):
        densityw_real = densityw_mean + np.random.normal(0, 1, densityw_mean.shape)* densityw_std
        tempw_real = tempw_mean + np.random.normal(0, 1, densityw_mean.shape)* tempw_std
        t0[ri], gamma[ri] = t0_gamma(densityw_real, tempw_real)
        
    return t0, gamma


@njit
def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0.0
    return correlation


@njit
def closest_argmin(A, B):
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]


def findNearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index, array[index]

def get_noise(flux, noise_model, flux_bins=None, bad=None, 
              flux_mean=None, flux_var=None):
        
    if isinstance(noise_model, np.float64):
        noise = np.full(flux.shape, noise_model)
    else:
        noise = np.zeros(flux.shape)
        #normalised the flux bins
        #flux_bins = (flux_bins - flux_mean)/flux_var
        for i in range(flux.shape[0]):
            #get the corresponding noise for each flux pixel
            noise[i] = noise_model[closest_argmin(flux[i], flux_bins)]
            #adjust the flux pixels to one, si they become featureless regions
            flux[i][bad==1] = 1.0
            #noise to zero in bad pixels regions

    return noise


@njit
def noise_model(flux, noise, bad):
    bin_size = 0.01
    bin_size_half = bin_size/2

    flux_bins = np.arange(-0.1, 1.1, bin_size)
    noise_med = np.zeros(len(flux_bins))
    noise_u1s = np.zeros(len(flux_bins))
    noise_l1s = np.zeros(len(flux_bins))
    
    flux_nonbad = flux[bad==0]
    noise_nonbad = noise[bad==0]
 
    for i in range(len(flux_bins)):
        samples = noise_nonbad[np.logical_and(
            flux_nonbad>=(flux_bins[i]-bin_size_half),
            flux_nonbad<=(flux_bins[i]+bin_size_half))]
        
        if len(samples) > 0:
            noise_l1s[i], noise_med[i], noise_u1s[i] = np.quantile(samples, [0.04, 0.5, 0.96])
        else:
            noise_l1s[i], noise_med[i], noise_u1s[i] = noise_l1s[i-1], noise_med[i-1], noise_u1s[i-1]
                
    return flux_bins, noise_med, noise_u1s, noise_l1s


def get_conf_cont(xx, yy):
    
    bins = 100
    #sigma = 2
    
    xmin = 1 #np.nanmin(xx)
    xmax = 30#np.nanmax(xx)
    
    ymin = 0.4#np.nanmin(yy)
    ymax = 2.2#np.nanmax(yy)

    #find bin sizes
    bin_x  = np.abs(xmax - xmin)/bins
    bin_y  = np.abs(ymax - ymin)/bins
        
    x = np.arange(xmin, xmax, bin_x)
    y = np.arange(ymin, ymax, bin_y)
    
    x_center = x[1:] + 0.5 * (x[1] - x[0])
    y_center = y[1:] + 0.5 * (y[1] - y[0])
    
    H, t1, t2 = np.histogram2d(xx.flatten(), yy.flatten(), bins=(x, y))
    H = H.T  #Let each row list bins with common y range  

    #H = gaussian_filter(H, .5)
     
    conf = find_conf_int(H)
    
    return x_center, y_center, H, conf


def project_temp_on_delta(density, temp, delta_bin=0.1):
    
    temp = temp.flatten()
    density = density.flatten()

    density_min = -1.1#np.min(density)
    density_max = .9#np.max(density)
    density_bins = np.arange(density_min, density_max, delta_bin)
    
    delta_bin_half = delta_bin/2
    temp_med = np.full(len(density_bins), np.nan)

    for i in range(len(density_bins)):
        temp_samples = temp[np.logical_and(density>=(density_bins[i]-delta_bin_half), 
                                            density<=(density_bins[i]+delta_bin_half))]
        
        if len(temp_samples) > 0:
            #mod introduces scatter
            #temp_med[i] = mode(np.around(temp_samples, 3)).mode[0]
            temp_med[i] = np.median(temp_samples)
    
    return density_bins, temp_med



#takes log values
@njit
def t0_gamma(density, temp):
    
    temp = temp.flatten()
    density = density.flatten()
    
    # #choose -0.7 as minmum density
    # #going lower biases gamma esimtate
    # #dont wanna go above 0.0 dominated
    # #by saturated pixels, biases estimates
    delta_bin = 0.1
    density_min = -0.4
    density_max = 0.2
    
    density_bins = np.arange(density_min, density_max, delta_bin)

    # #dont go lower than 0.05, too small bin
    # #max delta is -0.2 for predicted optically weighted data
    # #as forest is not probed densities lower than mean densities
    ##density_bins = np.array([-0.625, -0.375, -0.122, 0.125])
    ##delta_bin = 0.25
    
    delta_bin_half = delta_bin/2
    temp_med = np.full(len(density_bins), np.nan)

    for i in range(len(density_bins)):
        temp_samples = temp[np.logical_and(density>=(density_bins[i]-delta_bin_half), 
                                            density<=(density_bins[i]+delta_bin_half))]
        
        if len(temp_samples) > 1:
            temp_med[i] = np.median(temp_samples)
        
    # #select the bins which only has definite values
    density_bins = density_bins[np.isfinite(temp_med)]
    temp_med = temp_med[np.isfinite(temp_med)]
    p_coeffs = fit_poly(density_bins, temp_med, deg=1)
    
    #mask = np.logical_and(density>=-0.45, density<=0.25)
    #p_coeffs = fit_poly(density[mask], temp[mask], deg=1)
    
    t0 = eval_polynomial(p_coeffs, 0)
    gamma = p_coeffs[0] + 1
    
    #result = np.polyfit(density_bins, temp_med, 1)
    #t0 = np.poly1d(result)(0)
    #gamma = result[0] + 1

    return t0, gamma


def t0_gamma_to_sim_model(dir_dataset, t0, gamma, redshift):
    
    model_arr = [
        'planck1_20_1024',         'planck1_20_1024_cold',         'planck1_20_1024_hot',
        'planck1_20_1024_zr525',   'planck1_20_1024_zr525_cold',   'planck1_20_1024_zr525_hot',
        'planck1_20_1024_zr675',   'planck1_20_1024_zr675_cold',   'planck1_20_1024_zr675_hot',
        'planck1_20_1024_zr750',   'planck1_20_1024_zr750_cold',   'planck1_20_1024_zr750_hot',
        'planck1_20_1024_g10',     'planck1_20_1024_g14',          'planck1_20_1024_g16'
        ]
    
    # model_arr = [
    #     'planck1_20_1024_t04.2_gamma1.6',
    #      'planck1_20_1024_t04.0_gamma1.6',
    #      'planck1_20_1024_t03.9_gamma1.6',
    #     'planck1_20_1024_t04.2_gamma1.4',
    #      'planck1_20_1024_t04.0_gamma1.4',
    #      'planck1_20_1024_t03.9_gamma1.4',
    #     'planck1_20_1024_t04.2_gamma1.2',
    #      'planck1_20_1024_t04.0_gamma1.2',
    #      'planck1_20_1024_t03.9_gamma1.2',
        
    #       ]
    
    t0_arr = np.full(len(model_arr), np.nan)
    gamma_arr = np.full(len(model_arr), np.nan)
    distance_arr = np.full(len(model_arr), np.nan)
    
    prestr = "model_test_"
    
    for mi in range(len(model_arr)):
        post = prestr+model_arr[mi]+'_z'+"{:.2f}".format(redshift) + ".npy"
        filename = dir_dataset + post
        data = np.load(filename,'rb')
    
        t0_arr[mi], gamma_arr[mi] = t0_gamma(data["density"], data["temp"])
        distance_arr[mi] = np.sqrt( (10**t0_arr[mi]-10**t0)**2 + (gamma_arr[mi]-gamma)**2 )

    #find the closed grid model in the t0, 
    #gamma space, euclidean distance
    index = np.argmin(distance_arr)
    
    print(t0, gamma, redshift, index, model_arr[index], t0_arr[index], gamma_arr[index])

    return model_arr[index]
    
 
@njit
def pdf(data, bins, std=None):

    bin_size = (bins[1] - bins[0]) * 0.5
    hist, _ = np.histogram(data, bins=bins)
    pdf = hist/(np.sum(hist)*bin_size)
    hist_std = np.full(len(bins)-1, 0.0)
        
    #if the data has sigma for each data point
    if std is not None:
        
        #combine all data point in each histogram bin into one distribution, 
        #one mean, one sigma for each histogram bin
        inds = np.digitize(data, bins)
        
        for i in range(len(bins)-1):
            if len(std[inds==i])>0:
                #combined the variance in quaderature
                hist_std[i] = np.sqrt( np.sum(std[inds==i]**2) /np.float32(len(std[inds==i])) )
        
        return pdf, hist_std

    return pdf


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0.0
    return correlation


#find confidence intervals one sigma and 2 simga contours
#86th, 95th and 99th are 1/2/3sigma
#for 2D functions using CDF
def find_conf_int(H):
    Hc   = H.flatten()
    Hc   = np.sort(Hc)
    Hc   = Hc[::-1]
    
    Hcum = np.cumsum(Hc)
    Hcum/= Hcum[-1]
    return np.interp([.95, .68], Hcum, Hc)


#find confidence intervals for 1D functions using CDF
def conf_intv(data, percentile):
    counts, bin_edges = np.histogram(data)
    cdf = np.cumsum(counts)/float(np.sum(counts))
    bin_edges+= (bin_edges[1] - bin_edges[0])/2.0
    bin_edges = bin_edges[:len(bin_edges)-1]

    return np.interp(percentile, cdf, bin_edges)


#use this function for evaluating power spectrum
def get_pk_fft(data, vmax, factor):
    
    ##data = data/np.mean(data)- 1.0
    numlos = data.shape[0]
    nbins = data.shape[1]
    ps = np.zeros(nbins)
    dv = vmax/nbins
    
    for i in range(numlos):
        ps += np.abs(np.fft.fftshift(np.fft.fft(data[i,:])))**2
    
    k_array = np.fft.fftshift(np.fft.fftfreq(nbins, dv))
    
    k = 2*np.pi*k_array
    ps *= vmax/np.float32(numlos)
    ps = ps*k/np.pi/nbins*factor
    
    return np.log10(k), np.log10(ps)


def get_pdf(data):
    pdf, bins = np.histogram(data, bins=100, normed=True)
    bins_cen = bins[1:] + np.abs(bins[1] - bins[0]) * 0.5
    return bins_cen, pdf

def calc_curv(velaxis, flux):
    
    v = np.full(flux.shape[0], np.nan)
    
    for si in range(flux.shape[0]):
        dflux_dv   = np.gradient(flux[si], velaxis)
        d2flux_dv2 = np.gradient(dflux_dv, velaxis)
        v[si] = np.log10(np.mean(np.abs(d2flux_dv2 / np.power(1.+ dflux_dv**2, 1.5))))
        
    return np.mean(v)

#using redshift fits from Becker 2013
def taueff_z(redshift):

    #[τ0, β, C] = [0.751, 2.90, −0.132]
    #z0=3.5
    #best fit for taueff evolution see BECKER 2013
    tau0 = 0.751
    z0 = 3.5
    beta = 2.90
    C = -0.132
    
    return tau0 * np.power((1+redshift)/(1+z0), beta) + C

    
#read the Becker 2013 table from .dat file and return the nearest 
#mean flux and sigma from the nearest discrete bin
#use Sarah Bossman mean flux at z=>5
def mean_flux_z(redshift):
    
    data = np.loadtxt('becker2013.dat', usecols=(0, 5, 6), dtype=np.float32)
    redshift_index,_ = findNearest(data[:,0], redshift)
    
    #Bosman+2018, Table 4
    if redshift<5:
        mean_flux = data[redshift_index,1] #mean flux value for sigma
    elif redshift==5.0:
        mean_flux = 0.135
    elif redshift==5.3:
        mean_flux = 0.114

    return mean_flux
            

def gammaCurve(z, nH):
    
    ##Chardin+2018
    # redshifts = [3.0, 4.0, 5.0]
    # n0 = [0.0090, 0.0093, 0.0103]
    # alpha1 = [-1.12, -0.95, -1.29]
    # alpha2 = [-1.65, -1.50, -1.60]
    # beta = [5.32, 5.87, 5.06]
    # f = [0.018, 0.015, 0.024]

    # #Rahmati+2013
    redshifts = [3.0, 4.0, 5.0]
    n0 = [0.0074, 0.0058, 0.0044]
    alpha1 = [-1.99, -2.05, -2.63]
    alpha2 = [-0.88, -0.75, -0.57]
    beta = [1.72, 1.93, 1.77]
    f = [0.04, 0.02, 0.01]
    
    ind = findNearest(redshifts, z)
    
    print('nearest redshift', redshifts[ind])
        
    return (1-f[ind]) * np.power(1+np.power(nH/n0[ind], beta[ind]), alpha1[ind])\
        + f[ind] * np.power( (1 + nH/n0[ind]), alpha2[ind])


#Hubble function
def HubbleZ(z):
    return HUBBLE*100.*np.sqrt(OMEGAM*(1.+z)*(1.+z)*(1.+z) + (1.-OMEGAM))


def fwhm_to_bins(fwhm, skewer_length, z, hubble):    
    sigma = fwhm/(2*np.sqrt(2 * np.log(2)))
    vmax = skewer_length/(1+z)/hubble*HubbleZ(z)
    return  np.int32(np.floor(vmax/sigma))


def bins_to_fwhm(bins, skewer_length, z, hubble):    
    vmax = skewer_length/(1+z)/hubble*HubbleZ(z)
    return  vmax/np.float32(bins) * (2*np.sqrt(2 * np.log(2)))


def mpch_to_dz(skewer_length, z, hubble):
    vmax = skewer_length/(1+z)/hubble * HubbleZ(z)
    #vel_axis = np.arange(bins) * vmax/bins
    #z+1 = sqrt(1+v/c  / 1-v/c) * (1+z_em)
    #return  np.sqrt( (1+(vel_axis /speedOfLight)) / (1-(vel_axis /speedOfLight)) ) * (1+z) - 1
    return  (np.sqrt( (1+(vmax /speedOfLight)) / (1-(vmax /speedOfLight)) ) * (1+z) - 1) - z


def rebin(x,bin):
    return np.array([a.mean() for a in np.array_split(x,bin)])


def gamma_analytical (nH_bins, f, n0, alpha, beta):
    n0 = np.log10(n0)
    return (1-f) * np.power(1+np.power( np.power(10, nH_bins-n0), beta), alpha[0]) +\
 f * np.power(1 + np.power(10, nH_bins-n0), alpha[1])



def cumint(y, x):
    # Get the cumulative integral of y over x.
    integral = intg.cumtrapz(y, x)
    # Append zero to the front to include the integral between x[0] and itself.
    # This also makes sure the output array is equal in length to the input.
    integral = np.concatenate((np.array([0]), integral))

    return integral

def nH_global(z):
    a = 1. / (1. + z)
    rhoc_cgs = 1.87890e-29
    ob = 0.0482266
    om = 0.305147
    h0 = 0.68
    rhom_avg = rhoc_cgs * om * h0 ** 2 / a ** 3
    rhob_avg = rhom_avg * ob / om
    XHy = 0.7547
    mH_cgs = 1.67223e-24

    return rhob_avg * XHy / mH_cgs

def alphab(T):
    k_cgs = 1.38066e-16
    eV_cgs = 1.60218e-12
    K2eV = k_cgs / eV_cgs
    eV2K = 1.0 / K2eV
    xHI = 13.60 * eV2K / T
    x = 2 * xHI
    ans = (2.753e-14 * x ** 1.500) / (1.0 + (x / 2.740) ** 0.407) ** 2.2420

    return ans

#################################################################


def get_best_distribution(data):
    
    #dist_names =["gengamma","genhyperbolic","burr12"]
    #dist_names = _distn_names
    
    dist_names = [
        "norm", "lognorm", "gamma", "beta"
                  ]
    
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist.cdf, args=param)
        print( dist_name,", thres=", D, ", p=", p)
        print(param)
        dist_results.append((dist_name, p))

    #select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: ", best_p)
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]




def tau_r(r, nHI): 
    return -cumint(nHI*sigma_HI, r)

def gamma_r(gamma_bk, tau): 
    return gamma_bk*np.exp(-tau)
    

def nHI_r(r, gamma, nH): 
    A = -alphab(T)*(1+chi)
    B = gamma + 2.*alphab(T)*(1+chi)*nH
    C = -alphab(T)*(1.+chi)*nH**2.
    nHI = (-B + np.sqrt(B**2 - 4.*A*C))/2./A

    return nHI


def gamma_nH_curve ( r0, alpha, z, gamma_bk):
   
    n = 10000
    nH = np.zeros(n)
    gamma = gamma_bk*np.ones(n)
    r = np.logspace(2, -5, n)*KPC2CM/h
    print(r)
    
    itercount = 100
    nH = nH_global(z) * np.power( r/(r0*KPC2CM/h), -alpha)    
    
    for j in range(itercount):
                
        nHI = nHI_r(r, gamma, nH)
        tau = tau_r(r, nHI)
        gamma = gamma_r( gamma_bk, tau)

    return nH, gamma
    

#################################################################


def getPDF(data, bin_min, bin_max, bin_size):
    
    bins     = np.arange(bin_min, bin_max, bin_size)
    bins_cen = bins[1:] + bin_size/2
    mass_hist, _ = np.histogram(data, bins=bins)
    pdf = (mass_hist)/(np.sum(mass_hist)*bin_size)
        
    return bins_cen, pdf


    
def getPDFLogx(data, bin_min, bin_max, bin_size):
    
    bins = np.arange(bin_min, bin_max, bin_size)
    bins_cen = bins[1:] + bin_size/2
    
    mass_hist, _ = np.histogram(np.log10(data), bins=bins)
    pdf = mass_hist/(np.sum(mass_hist)*bin_size)
    
    return bins_cen, pdf
     


def readSystems(file_dir, zs):
            
    dt = np.dtype([('x', np.int32), ('y', np.int32), ('z', np.int32),  
                   ('mass', np.float), ('volume', np.float), \
                   ('NHI', np.float), ('delta', np.float), \
                   ('temp', np.float),('xHI_delta', np.float),\
                   ('gamma', np.float), ('ratio_max', np.float)
                   ])
            

    file  = file_dir+'/sys_3D_vthres_z='+zs
    sys_3d = np.fromfile(file, dtype=dt)
                
    file  = file_dir+'/dm_mass_vthres_z='+ zs
    dm_mass_sys = np.fromfile(file, dtype=np.float)
    
    ratio_v = 1.5
            
    #it is the total size in kpc/h**3
    vol_sys = sys_3d['volume']
            
    #find the corresponding diameter if it is a sphere with volume 4/3pi r**3
    #units in kpc/h, comoving
    size_sys = np.cbrt(vol_sys * (3/4) / np.pi) * 2 
            
    #gas masses in solar convert it to total
    gas_mass_sys = sys_3d['mass']
    
    ratio_max = sys_3d['ratio_max']
    x = sys_3d['x']
    y = sys_3d['y']
    z = sys_3d['z']
        
    
    #choosing systems with some minimum volume
    #select only systems with certain pixels size
    min_volume = 2
    
    #min_req =  np.cbrt(min_volume * (3/4) / np.pi) * 2
    #print("Minimum Volume [kpc/h]^3 = ", min_volume,", deq[kpc/h]=", min_req)
    
    #add DM mass to gas mass
    mass_sys = dm_mass_sys + gas_mass_sys
    
    est_mass_sys = gas_mass_sys + gas_mass_sys * ( (OMEGAM-OMEGAB)/OMEGAB)
    
    gas_frac_sys = gas_mass_sys / mass_sys
    est_gas_frac_sys = OMEGAB/(OMEGAM-OMEGAB)
    
    #X,Y,Z as tuple
    #coords = list(map(tuple,np.dstack((x,y,z)).reshape(-1,3)))
    #coords_red = list(map(tuple,np.dstack((x[ratio_max>ratio_v], y[ratio_max>ratio_v], z[ratio_max>ratio_v])).reshape(-1,3)))
    #coords_blue = list(map(tuple,np.dstack((x[ratio_max<=ratio_v], y[ratio_max<=ratio_v], z[ratio_max<=ratio_v])).reshape(-1,3)))


    return mass_sys[vol_sys>=min_volume], est_mass_sys[vol_sys>=min_volume], size_sys[vol_sys>=min_volume], \
 gas_frac_sys[vol_sys>=min_volume], est_gas_frac_sys, \
sys_3d['NHI'][vol_sys>=min_volume], sys_3d['delta'][vol_sys>=min_volume],\
 sys_3d['temp'][vol_sys>=min_volume], sys_3d['xHI_delta'][vol_sys>=min_volume], \
 sys_3d['gamma'][vol_sys>=min_volume], sys_3d['ratio_max'][vol_sys>=min_volume]#, coords, coords_red, coords_blue

#################################################################

def cube2sk(file, data):
   
    sk_pars = np.fromfile(file, dtype=np.float).reshape( numlos, 6)
    start_pos = sk_pars[:,:3]
    nvec      = sk_pars[:,3:]
      
    sk = np.full( (numlos,bins), -1 )   
    
    #loop over skewers, set by N variable
    for i in range(numlos):
        for j in range(bins):
        
            #get coordinates from start position and units vector
            coords = nvec[i]*j*dxh + start_pos[i] 
 			
            #the staring index of current skewer NGP
            index =  ( ( coords/dxh +.5 + 1000*Ngas)%Ngas ).astype(int)
            sk[i,j] = data[index[0],index[1],index[2]]
    
    
    return  sk


def mfp2NHI(mfp):
    return scale/mfp/thompson

def NHI2taueff(NHI):
    return NHI*thompson

def taueff2NHI(taueff):
    return taueff/thompson

def taueff2mfp(taueff):
    return scale/taueff


#alpahB is the total recombination coefficient minus alphaA contributions from recombinations directly into the
#ground state. This form of the photoionization equilibrium condition means that the following conditions 
#will prevail in an optically thick nebula:
#1. Photoionization by the stellar radiation field is balanced by recombination into excited states of H.
#2. Recombinations directly into the 1s 2S ground state emit ionizing photons that are quickly reabsorbed 
#by the nebula, and so have no net effect on the overall ionization balance.
def alphaB(T):
    x = 2. * 157807. / T
    return 2.753e-14*np.power(x,1.500)/np.power((1.0+pow((x/2.740),0.407)),2.2420)


#recombination rates in units of [cm^3 s^-1], recombinations dierently into ground state of 1S hydrogen
def alphaA(T):
	lambdaHI = 2. * 157807. / T
	#Hui and Gendin 1997
	return ( 1.269e-13*np.power(lambdaHI,1.503) )/np.power( 1. + np.power(lambdaHI/0.522,0.470) ,1.923);
	#return 4.063e-13*np.power(T/10000,-0.72);