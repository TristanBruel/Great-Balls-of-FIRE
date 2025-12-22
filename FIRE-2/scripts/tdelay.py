import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os


def plot_mBH1_dist():

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    #plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#e41a1c']

    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))

    #tbins = np.linspace(0, 14, 57)
    tbins = np.logspace(-3, 2, 51)

    bbhfile = os.path.join(FIRE_dir, 'grid/allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    tform, m1, m2, tmerge = np.loadtxt(bbhfile, usecols=[2,4,5,7], skiprows=1).T
    m1 = np.maximum(m1,m2)
    tdelay = tmerge - tform
    
    for n,simu in enumerate(['m12i','m11d','m11h','m11e']):
        in_sim = np.array([c.split('_')[0]==simu for c in cluster_id])
        hist, _ = np.histogram(tdelay[in_sim], bins=tbins)
        dN = hist /np.diff(tbins)
        ax.stairs(dN, tbins,
                  color=CB_color_cycle[n], lw=lw, label=simu,
            )

    ## Time delays of massive BBH mergers ##
    massive = m1 >= 50
    hist, _ = np.histogram(tdelay[massive], bins=tbins)
    dN = hist /np.diff(tbins)
    ax.stairs(dN, tbins,
              color='k', lw=lw, hatch='//',
              label=r'$M_\mathrm{BH1}>50\ \mathrm{M}_\odot$',
        )

    ## Add a 1/t distribution as a comparison ##
    powerlaw = 5e2 *1/tbins
    ax.plot(tbins[tbins>1e-2], powerlaw[tbins>1e-2], 
            color='k', lw=lw-0.5, linestyle='--',
            label=r'$\mathrm{dN}/\mathrm{d}\tau\propto 1/\tau$',
            )


    #ax.set_xlabel(r'$\tau\ [\mathrm{Gyr}]$', fontsize=fs)
    ax.set_xlabel(r'$\tau$', fontsize=fs)
    #ax.set_xlim(xmin=0., xmax=14)
    #ax.set_xticks(np.linspace(0,12.5,6))
    ax.set_xscale('log')
    ax.set_xlim(xmin=1e-3, xmax=20)
    ax.set_xticks([1e-3,1e-2,1e-1,1,10],['1 Myr', '10 Myr', '100 Myr', '1 Gyr', '10 Gyr'])
    ax.set_ylabel(r'$\mathrm{dN}/\mathrm{d}\tau\ [\mathrm{Gyr}^{-1}]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e0, ymax=5e5)
    ax.set_box_aspect(1)
    ax.legend(loc='upper right', fontsize=fs, ncols=2)

    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'tdelays.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    #FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'
    FIRE_dir = '/home/tbruel/Great-Balls-of-FIRE/FIRE-2'

    plot_mBH1_dist()
    plt.show()
