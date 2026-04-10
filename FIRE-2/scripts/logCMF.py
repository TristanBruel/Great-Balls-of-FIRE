import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import os

def func_linear(x, A, B):
    return A + B*x

def func_powerlaw(x, A, B, alpha):
    return A + B*x**alpha


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
    midbins = (logmbins[1:]+logmbins[:-1])/2
    ntot = 0
    
    ## Start with m12i, the MHD version of GBoF I & II ##
    sim_dir = os.path.join(FIRE_dir,'m12i_MHD_res7100')
    cluster_file = os.path.join(sim_dir, 'ClusterPopulation_1_reduced.hdf5')
    with h5py.File(cluster_file) as c:
        mass = c['Mass'][()]
    print('Number of massive star clusters:', np.sum(mass>6e4))
    Ncl, _ = np.histogram(np.log10(mass), bins=logmbins)
    Ncl = Ncl /np.diff(logmbins)
    ax.stairs(np.log10(Ncl), logmbins,
            color=CB_color_cycle[0], lw=lw, label='m12i',
            )

    # Best-fit power-law #
    popt, pcov = curve_fit(func_linear, midbins[Ncl>0], np.log10(Ncl[Ncl>0]))
    ax.plot(logmbins, func_linear(logmbins, *popt),
            color=CB_color_cycle[0], ls='--',
            )
    print('Best-fit powerlaw dN/dMcl = A * Mcl^alpha is: alpha=%.2f' %(popt[1]-1))

    ## Dwarf galaxies ##
    for n,simu in enumerate(['m11d','m11h','m11e','m11q','m11i']):
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
                color=CB_color_cycle[n+1], lw=lw, label=simu,
                )

        ## ADD BEST FIT POWER LAW MODEL ##
        # Best-fit power-law #
        popt, pcov = curve_fit(func_linear, midbins[Ncl>0], np.log10(Ncl[Ncl>0]))
        ax.plot(logmbins, func_linear(logmbins, *popt),
                color=CB_color_cycle[n+1], ls='--',
                )
        print('Best-fit powerlaw dN/dMcl = A * Mcl^alpha is: alpha=%.2f\n' %(popt[1]-1))


    ax.fill_between(logmbins[logmbins<np.log10(6e4)], y1=1, y2=8,
                    facecolor='gray', alpha=.5, zorder=10,
                    )


    ax.set_xlabel(r'$\mathrm{log}_{10}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=3, xmax=7)
    ax.set_xticks(np.linspace(3,7,5))
    ax.set_ylabel(r'$\mathrm{log}_{10}(\mathrm{dN}/\mathrm{dlog}_{10}(M_\mathrm{cl}/\mathrm{M}_\odot))$', fontsize=fs)
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

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    plot_CMF()
    plt.show()
