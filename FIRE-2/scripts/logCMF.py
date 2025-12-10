import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

    logmbins = np.linspace(3, 7, 41)
    ntot = 0

    for n,simu in enumerate(['m12i','m11d','m11h','m11e','m11q','m11i']):
        sim_dir = os.path.join(FIRE_dir,simu+'_res7100')
        print(sim_dir)

        cluster_file = os.path.join(sim_dir, 'AllClusters_'+simu+'.hdf5')
        with h5py.File(cluster_file) as c:
            mass = c['Mass'][()]
        print('Number of massive star clusters:', np.sum(mass>6e4))
        ntot += len(mass)
        Ncl, _ = np.histogram(np.log10(mass), bins=logmbins)
        Ncl = Ncl /np.diff(logmbins)
        ax.stairs(np.log10(Ncl), logmbins,
                color=CB_color_cycle[n], lw=lw, label=simu,
                )

        ## ADD BEST FIT POWER LAW MODEL ##

    MHD_dir = os.path.join(FIRE_dir, 'm12i_MHD_res7100')
    print(MHD_dir)
    cluster_file = os.path.join(MHD_dir, 'ClusterPopulation_1_reduced.hdf5')
    with h5py.File(cluster_file) as c:
        mass = c['Mass'][()]
    print('Number of massive star clusters:', np.sum(mass>6e4))
    Ncl, _ = np.histogram(np.log10(mass), bins=logmbins)
    Ncl = Ncl /np.diff(logmbins)
    ax.stairs(np.log10(Ncl), logmbins,
            color=CB_color_cycle[n+1], lw=lw, label='mhd',
            )

    print('Total number of clusters:', ntot)
    ax.fill_between(logmbins[logmbins<np.log10(6e4)], y1=1, y2=8,
                    facecolor='gray', alpha=.5,
                    )

    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=3, xmax=7)
    ax.set_xticks(np.linspace(3,7,5))
    ax.set_ylabel(r'$\mathrm{log}(\mathrm{dN}/\mathrm{dlog}(M_\mathrm{cl}/\mathrm{M}_\odot))$', fontsize=fs)
    ax.set_ylim(ymin=1, ymax=7.5)
    ax.set_box_aspect(1)
    ax.legend(loc='upper right', fontsize=fs)

    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'logCMF.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'

    plot_CMF()
    plt.show()
