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
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#e41a1c']

    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))

    mbins = np.linspace(0, 250, 51)

    ## Use the m12i MHD version of GBoF I & II ##
    sim_dir = os.path.join(FIRE_dir,'m12i_MHD_res7100')
    bbhfile = os.path.join(sim_dir,'bh_dyn.dat')
    m1, m2, weights = np.loadtxt(bbhfile, usecols=[4,5,12], skiprows=1).T
    m1 = np.maximum(m1, m2)
    hist, _ = np.histogram(m1, bins=mbins, weights=weights)
    dN = hist /np.diff(mbins)
    ax.stairs(dN, mbins,
              color=CB_color_cycle[0], lw=lw, label='m12i',
        )

    ## Other FIRE-2 simulations
    bbhfile = os.path.join(FIRE_dir, 'grid/allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2 = np.loadtxt(bbhfile, usecols=[4,5], skiprows=1).T
    m1 = np.maximum(m1, m2)
    
    for n,simu in enumerate(['m11d','m11h','m11e']):
        in_sim = np.array([c.split('_')[0]==simu for c in cluster_id])
        hist, _ = np.histogram(m1[in_sim], bins=mbins)
        dN = hist /np.diff(mbins)
        ax.stairs(dN, mbins,
                  color=CB_color_cycle[n+1], lw=lw, label=simu,
            )


    ax.set_xlabel(r'$M_\mathrm{BH1}\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(xmin=0., xmax=210)
    ax.set_xticks(np.linspace(0,200,5))
    ax.set_ylabel(r'$\mathrm{dN}/\mathrm{d}M_\mathrm{BH1}\ [\mathrm{M}_\odot^{-1}]$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-1, ymax=2e3)
    ax.set_box_aspect(1)
    ax.legend(loc='upper right', fontsize=fs)

    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'mBH1.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    plot_mBH1_dist()
    plt.show()
