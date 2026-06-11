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

    ebins = np.linspace(0., 1., 51)

    ## Use the m12i MHD version of GBoF I & II ##
    print('m12i_MHD')
    sim_dir = os.path.join(FIRE_dir,'m12i_MHD_res7100')
    bbhfile = os.path.join(sim_dir,'bh_dyn.dat')
    ecc, weights = np.loadtxt(bbhfile, usecols=[6,12], skiprows=1).T
    print('Number of BBH mergers:',len(ecc))
    print('Number of BBH mergers with e10Hz > 0.2:',np.sum(ecc>0.2))
    hist, _ = np.histogram(ecc, bins=ebins, weights=weights)
    ax.stairs(hist, ebins,
              color=CB_color_cycle[0], lw=lw, label='m12i',
        )

    ## Other FIRE-2 simulations
    print('FIRE-2 dwarfs')
    bbhfile = os.path.join(FIRE_dir, 'grid/allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    ecc = np.loadtxt(bbhfile, usecols=6, skiprows=1)
    print('Number of BBH mergers:',len(ecc))
    print('Number of BBH mergers with e10Hz > 0.2:',np.sum(ecc>0.2))
    
    for n,simu in enumerate(['m11d','m11h','m11e']):
        in_sim = np.array([c.split('_')[0]==simu for c in cluster_id])
        hist, _ = np.histogram(ecc[in_sim], bins=ebins)
        ax.stairs(hist, ebins,
                  color=CB_color_cycle[n+1], lw=lw, label=simu,
            )


    ax.set_xlabel(r'$e_\mathrm{10 Hz}$', fontsize=fs)
    ax.set_xlim(xmin=0., xmax=1.)
    #ax.set_xscale('log')
    ax.set_xticks(np.linspace(0,1,5))
    ax.set_ylabel(r'$\mathrm{N}_\mathrm{BBHs}$', fontsize=fs)
    ax.set_yscale('log')
    #ax.set_ylim(ymin=1e-1, ymax=2e3)
    ax.set_box_aspect(1)
    ax.legend(loc='upper right', fontsize=fs)

    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    #plt.savefig(os.path.join(plot_dir,'ecc.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    plot_mBH1_dist()
    plt.show()
