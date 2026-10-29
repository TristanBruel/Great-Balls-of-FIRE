import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os


def plot_loge_dist():

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#e41a1c']

    fig, axes = plt.subplots(1, 2, figsize=(8.,3.5))

    loge_bins = np.linspace(-8, 0, 41)

    ## Use the m12i MHD version of GBoF I & II ##
    print('m12i_MHD')
    sim_dir = os.path.join(FIRE_dir,'m12i_MHD_res7100')
    bbhfile = os.path.join(sim_dir,'bh_dyn.dat')
    teject, ecc, tmerge, weights = np.loadtxt(bbhfile, usecols=[3,6,7,12], skiprows=1).T
    ejected = teject > 0

    # All BBH mergers over cosmic time #
    ax = axes[0]
    print('Total number of BBH mergers:', len(ecc))
    print('Number of ejected BBH mergers:', np.sum(ejected))

    hist_incluster, _ = np.histogram(np.log10(ecc[~ejected]), bins=loge_bins, 
                                     weights=weights[~ejected],
                                     )
    ax.stairs(hist_incluster, loge_bins,
              color=CB_color_cycle[0], lw=lw, label='in-cluster',
        )

    hist_ejected, _ = np.histogram(np.log10(ecc[ejected]), bins=loge_bins, 
                                   weights=weights[ejected],
                                   )
    ax.stairs(hist_ejected, loge_bins,
              color=CB_color_cycle[1], lw=lw, label='ejected',
        )


    ax.set_xlabel(r'$\mathrm{log}_{10}(e/\mathrm{10 Hz})$', fontsize=fs)
    #ax.set_xlim(xmin=0., xmax=1.)
    #ax.set_xscale('log')
    ax.set_xticks(np.linspace(-8,0,9))
    ax.set_ylabel(r'$\mathrm{N}_\mathrm{BBHs}$', fontsize=fs)
    #ax.set_yscale('log')
    #ax.set_ylim(ymin=1e-1, ymax=2e3)
    ax.set_box_aspect(1)
    ax.set_title(r'All mergers across cosmic time (%i)' %np.sum(weights))
    ax.legend(loc='upper right', fontsize=fs)


    # BBH mergers in the last 100 Myr #
    ax = axes[1]
    snapshot_file = os.path.join(sim_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
    tmax = times.max()
    local = tmax-tmerge < 0.1
    print('Number of BBH mergers in the last 100 Myr of the cosmo sim:', np.sum(local))
    print('Number of ejected mergers:', np.sum(ejected&local))

    hist_incluster, _ = np.histogram(np.log10(ecc[local&~ejected]), bins=loge_bins,
                                     weights=weights[local&~ejected],
                                     )
    ax.stairs(hist_incluster, loge_bins,
              color=CB_color_cycle[0], lw=lw, label='in-cluster',
        )

    hist_ejected, _ = np.histogram(np.log10(ecc[local&ejected]), bins=loge_bins,
                                   weights=weights[local&ejected],
                                   )
    ax.stairs(hist_ejected, loge_bins,
              color=CB_color_cycle[1], lw=lw, label='ejected',
        )


    ax.set_xlabel(r'$\mathrm{log}_{10}(e/\mathrm{10 Hz})$', fontsize=fs)
    #ax.set_xlim(xmin=0., xmax=1.)
    #ax.set_xscale('log')
    ax.set_xticks(np.linspace(-8,0,9))
    ax.set_ylabel(r'$\mathrm{N}_\mathrm{BBHs}$', fontsize=fs)
    #ax.set_yscale('log')
    #ax.set_ylim(ymin=1e-1, ymax=2e3)
    ax.set_box_aspect(1)
    ax.set_title(r'Mergers in the last 100 Myr (%i)' %np.sum(weights[local]))
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

    plot_loge_dist()
    plt.show()
