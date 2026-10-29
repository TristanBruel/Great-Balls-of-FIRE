import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import os


def plot_CMF():

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#e41a1c',
                      '#984ea3', '#999999', '#dede00']

    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))

    logrbins = np.linspace(-2, 2, 41)
    midbins = (logrbins[1:]+logrbins[:-1])/2
    ntot = 0
    
    ## Start with m12i, the MHD version of GBoF I & II ##
    sim_dir = os.path.join(FIRE_dir,'m12i_MHD_res7100')
    cluster_file = os.path.join(sim_dir, 'ClusterPopulation_1_reduced.hdf5')
    with h5py.File(cluster_file) as c:
        mass = c['Mass'][()]
        radius = c['HalfMassRadius'][()]
    print('Number of massive star clusters:', np.sum(mass>6e4))
    Ncl, _ = np.histogram(np.log10(radius), bins=logrbins)
    Ncl = Ncl /np.diff(logrbins)
    ax.stairs(np.log10(Ncl), logrbins,
            color=CB_color_cycle[0], lw=lw, label='m12i',
            )


    ## Dwarf galaxies ##
    for n,simu in enumerate(['m11d','m11h','m11e','m11q','m11i']):
        sim_dir = os.path.join(FIRE_dir,simu+'_res7100')
        print(sim_dir)

        cluster_file = os.path.join(sim_dir, 'AllClusters_'+simu+'.hdf5')
        with h5py.File(cluster_file) as c:
            mass = c['Mass'][()]
            radius = c['HalfMassRadius'][()]
        print('Number of massive star clusters:', np.sum(mass>6e4))
        ntot += len(mass)
        Ncl, _ = np.histogram(np.log10(radius), bins=logrbins)
        Ncl = Ncl /np.diff(logrbins)
        ax.stairs(np.log10(Ncl), logrbins,
                color=CB_color_cycle[n+1], lw=lw, label=simu,
                )



    ax.set_xlabel(r'$\mathrm{log}_{10}(r_\mathrm{hm}/\mathrm{pc})$', fontsize=fs)
    ax.set_xlim(xmin=-2, xmax=2)
    ax.set_xticks(np.linspace(-2,2,5))
    ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{dN}/\mathrm{dlog}_{10}(r_\mathrm{hm}/\mathrm{pc}))$', fontsize=fs)
    #ax.set_ylim(ymin=1, ymax=7.5)
    ax.set_box_aspect(1)
    ax.legend(loc='lower left', fontsize=fs, ncols=2)

    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'logCRF.png'), bbox_inches='tight', dpi=1200)
    plt.savefig(os.path.join(plot_dir,'logCRF.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    plot_CMF()
    plt.show()
