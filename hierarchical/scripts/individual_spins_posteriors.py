import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

import os


def plot_spin_posteriors():

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))

    chibins = np.linspace(-1,1,101)

    ### Load LVK posteriors ###
    chi1_file = os.path.join(posterior_dir, 'chi1s.npy')
    chi1 = np.load(chi1_file)
    n_events = np.load(os.path.join(posterior_dir,'nevents.npy'))[0]
    n_samples = np.load(os.path.join(posterior_dir,'nsamples.npy'))[0]
    #for n in range(n_events):
    for n in range(1):
        chi_samples = chi1[(n*n_samples):((n+1)*n_samples)]
        hist, _ = np.histogram(chi_samples, bins=chibins, density=True)
        ax.stairs(hist, chibins, color='gray', alpha=.5)

    ax.set_xlabel(r'$\chi$', fontsize=fs)
    ax.set_ylabel(r'$p(\chi)$', fontsize=fs)

    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'individual_spin_posteriors.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    posterior_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/Alex/lvc_data_lvc_samples_full'

    plot_spin_posteriors()
    plt.show()
