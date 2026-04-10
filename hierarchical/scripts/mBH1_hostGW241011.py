import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

import os


def plot_mBH1():


    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    fig, ax = plt.subplots(1, 1, figsize=(5.,5.))
    
    m1bins = np.linspace(0,200,51)

    ### Show clusters that host GW241011-like BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs_withKremer.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1

    # BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses
    GW24x = (m1 < 30) & (q < 0.85) & (a1 > 0.05)
    unique, indexes = np.unique(cluster_id[GW24x], return_index=True)
    print('Looking for BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses')
    print('Number of events found:', np.sum(GW24x))
    print('Number of clusters hosting these events:', len(unique))
    #mBH = np.array([])
    #for i in range(len(unique)):
    #    mBH = np.append(mBH,m1[cluster_id==unique[i]])
    #print('Total number of BBH mergers from these clusters:',len(mBH),'\n')
    hist, _ = np.histogram(m1, bins=m1bins)
    ax.stairs(hist, m1bins,
              color='firebrick', label='all BBH mergers from CMC sims',
              )
    hist, _ = np.histogram(m1[GW24x], bins=m1bins)
    ax.stairs(hist, m1bins,
              color='firebrick', fill=True, label='GW241x1x-like mergers',
              )


    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(xmin=0, xmax=170)
    xticks = np.linspace(0,150,7)
    ax.set_xticks(xticks)
    ax.set_ylabel(r'$\mathrm{N}_\mathrm{BBHs}$', fontsize=fs)
    ax.set_ylim(ymin=0.8, ymax=5e3)
    ax.set_yscale('log')
    ax.set_box_aspect(1)
    ax.legend(loc='upper right', fontsize=fs)


    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'mBH1_host_GW241011.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'

    GW24_dir = '/home/tristan-bruel/Documents/Science/o4a-astro/GW241011_GW241110_data_release/data'
    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical'
    gw241011_color = '#1f78b4'
    gw241110_color = '#33a02c'

    plot_mBH1()
    plt.show()
