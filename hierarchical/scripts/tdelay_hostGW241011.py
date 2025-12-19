import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

import os


def plot_tdelay():


    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    fig, ax = plt.subplots(1, 1, figsize=(5.,5.))
    
    tmax = 14
    deltat = .250 # width of the time bins (in Gyr)
    nbins = round(tmax /deltat) + 1
    tbins = np.linspace(0,tmax,nbins)
    tbins = np.logspace(-3,2,51)

    ### Show clusters that host GW241011-like BBH mergers ###
    #bbhfile = os.path.join(grid_dir, 'allBBHs.dat')
    bbhfile = os.path.join(grid_dir, 'allBBHs_withKremer.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    tform, m1, m2, tmerge, a1, a2 = np.loadtxt(bbhfile, usecols=[2,4,5,7,8,9], skiprows=1).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1
    tdelay = tmerge-tform
    
    # BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses
    GW24x = (m1 < 30) & (q < 0.85) & (a1 > 0.05)
    unique, indexes = np.unique(cluster_id[GW24x], return_index=True)
    print('Looking for BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses')
    print('Number of events found:', np.sum(GW24x))
    print('Number of clusters hosting these events:', len(unique))
    #td = np.array([])
    #for i in range(len(unique)):
    #    td = np.append(td,tdelay[cluster_id==unique[i]])
    #print('Total number of BBH mergers from these clusters:',len(td),'\n')
    hist, _ = np.histogram(tdelay, bins=tbins)
    ax.stairs(hist, tbins,
              color='firebrick', label='all BBH mergers from CMC sims',
              )
    hist, _ = np.histogram(tdelay[GW24x], bins=tbins)
    ax.stairs(hist, tbins,
              color='firebrick', fill=True, label='GW241x1x-like mergers',
              )


    ax.set_xlabel(r'$t_\mathrm{delay}$', fontsize=fs)
    ax.set_xlim(xmin=1e-3, xmax=2e1)
    ax.set_xscale('log')
    ax.set_xticks([1e-3,1e-2,1e-1,1,10],['1 Myr','10 Myr','100 Myr','1 Gyr','10 Gyr'])
    ax.set_ylabel(r'$\mathrm{N}_\mathrm{BBHs}$', fontsize=fs)
    ax.set_yscale('log')
    #ax.set_ylim(ymin=1e-3, ymax=1e1)
    ax.set_box_aspect(1)
    ax.legend(loc='upper left', fontsize=fs)


    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'tdelay_host_GW241011.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    grid_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2/grid'

    plot_tdelay()
    plt.show()
