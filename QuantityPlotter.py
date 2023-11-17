import matplotlib.pyplot as plt
import numpy as np
import matplotlib


class QuantityPlotter:
    def __init__(self, rows, cols, width, height, skewer_length, hubble):

        self.skewer_length = skewer_length
        self.hubble = hubble
        self.rows = rows
        self.cols = cols
        self.height = height
        self.width = width
        self.set_plot()
        
    def set_dataset(self, flux, densityw, mean, upper, lower):
        self.flux = flux
        self.densityw = densityw
        self.mean = mean
        self.upper = upper
        self.lower = lower
        self.pixels = self.flux.shape[1]
        self.axis = np.arange(self.pixels) * self.skewer_length/self.pixels
                        
        
    def set_plot(self):
        
        font = {'family' : 'serif', 'weight' : 'normal','size' : 32}
        matplotlib.rc('font', **font)
        
        self.fig, self.ax = plt.subplots(
            self.rows, self.cols, figsize=(self.width*self.cols, 
                                           self.height*self.rows))
        self.fig.subplots_adjust(wspace=0, hspace=0)
    
    def adjust_plot(self):
        for i, ax in enumerate(self.ax):
            ax.tick_params(which='both',direction="in", width=1.5)
            ax.tick_params(which='major',length=14, top=True, left=True, 
                           right=True)
            ax.tick_params(which='minor',length=10, top=True, left=True, 
                           right=True)
            ax.minorticks_on()

    def save_plot(self, filename):   
        # Check if the filename ends with '.npy' and remove it
        if filename.endswith('.npy'):
            filename_pdf = filename[:-4][:]
             
        self.fig.savefig(filename_pdf+'.pdf', format='pdf', dpi=90, 
                         bbox_inches = 'tight')
     
     
    def close_figures(self):
        plt.close(self.fig)
        #plt.clf()
        
    def clear_curves(self):
        ax_flat = np.ravel(self.ax)
        for ax in ax_flat:
                ax.clear()
       
    def plot_quantity(self, axis, xx, yy, upper=None, lower=None, xlabel='', 
                      ylabel='', color="black",
                      color_shaded="red", xlim=None, ylim=None):
        if xlim is None:
            xlim = [np.min(xx),  np.max(xx)]
                
        # if ylim is None:
        #     if upper is None:
        #         ylim = [np.min(lower),  np.max(upper)]
        #     else:
        #         ylim = [np.min([yy, lower]),  np.max([yy, upper])]
        
        axis.step(xx, yy, where='mid', color=color, alpha=1)
        if xlim is not None:
            axis.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            axis.set_ylim(ylim[0], ylim[1])
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
                
        if xlabel=='':
            axis.set_xticklabels([])
        if ylabel=='':
            axis.set_yticklabels([])
        
        if ((upper is not None) and 
            (lower is not None)):
            axis.fill_between(xx, upper, y2=lower, color=color_shaded, 
                              alpha=.2)

    
    def plot_los(self, los_index):
        self.adjust_plot()
        
        self.plot_quantity(self.ax[0], self.axis, self.flux[los_index], 
                           xlabel='', ylabel=r'${\rm Flux}$', color="black")
        self.plot_quantity(self.ax[1], self.axis, self.densityw[los_index], 
                            xlabel='', ylabel=r'${\rm log}\Delta_{\rm \tau}$', 
                            color="black")
        self.plot_quantity(self.ax[1], self.axis, self.mean[los_index], 
                            self.upper[los_index], self.lower[los_index], 
                            xlabel=r'${\rm Mpc/{\rm h}}$', 
                            ylabel=r'${\rm log}\Delta_{\rm \tau}$', color="red")
        self.close_figures()
        
