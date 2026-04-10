import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os


def plot_Zdist(Zsun=0.017):

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

    logZbins = np.linspace(-4, 1, 51)

    ## Use the m12i MHD version of GBoF I & II ##
    sim_dir = os.path.join(FIRE_dir,'m12i_MHD_res7100')
    ## All star clusters ##
    cluster_file = os.path.join(sim_dir, 'ClusterPopulation_1_reduced.hdf5')
    with h5py.File(cluster_file) as c:
        mass = c['Mass'][()]
        metal = c['Metallicity'][()]

    Mcl, _ = np.histogram(np.log10(metal/Zsun), bins=logZbins, weights=mass)
    Mcl = Mcl /np.diff(logZbins) /np.sum(Mcl)
    ax.stairs(Mcl, logZbins,
              color=CB_color_cycle[0], lw=lw, label='m12i',
              )

    ## Other FIRE-2 simulations
    for n,simu in enumerate(['m11d','m11h','m11e','m11q','m11i']):
        sim_dir = os.path.join(FIRE_dir,simu+'_res7100')
        cluster_file = os.path.join(sim_dir, 'AllClusters_%s.hdf5' %simu)
        with h5py.File(cluster_file) as c:
            mass = c['Mass'][()]
            metal = c['Metallicity'][()]

        #tokeep = mass>=1e4
        #mass = mass[tokeep]
        #metal = metal[tokeep]
        Mcl, _ = np.histogram(np.log10(metal/Zsun), bins=logZbins, weights=mass)
        Mcl = Mcl /np.diff(logZbins) /np.sum(Mcl)
        ax.stairs(Mcl, logZbins,
                  color=CB_color_cycle[n+1], lw=lw, label=simu,
              )


    ax.set_xlabel(r'$\mathrm{log}_{10}(Z/\mathrm{Z}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=-4.2, xmax=1.2)
    ax.set_xticks(np.linspace(-4,1,6))
    ax.set_ylabel(r'$\mathrm{dn}/\mathrm{dlog}_{10}(Z/\mathrm{Z}_\odot)$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-3, ymax=1e1)
    ax.set_box_aspect(1)
    ax.legend(loc='upper left', fontsize=fs)

    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'Zdist_allsims.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    plot_Zdist()
    plt.show()
