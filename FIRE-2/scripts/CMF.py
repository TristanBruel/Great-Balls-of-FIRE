import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
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
                  '#f781bf', '#a65628', '#e41a1c']
    cmap = mpl.colormaps['viridis']

    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))

    logmbins = np.linspace(3, 8, 51)
    midbins = (logmbins[1:]+logmbins[:-1])/2
    ntot = 0
    #for n,simu in enumerate(['m12i','m11d','m11h','m11e','m11q','m11i']):
    for n,simu in enumerate(['m12i']):
        sim_dir = os.path.join(FIRE_dir,simu+'_res7100')
        snapshot_file = os.path.join(sim_dir, 'snapshot_times.txt')
        redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T
        tmax = times.max()
        deltat = 250 # width of the time bins (in Myr)
        nbins = round(1e3*tmax /deltat) + 1
        tbins = np.linspace(0, tmax, nbins)
        colors = cmap(np.linspace(0, 1, len(tbins)-1))

        cluster_file = os.path.join(sim_dir, 'AllClusters_'+simu+'.hdf5')
        with h5py.File(cluster_file) as c:
            mass = c['Mass'][()]
            tform = c['FormationTime'][()]
        ntot += len(mass)
        for t in range(len(tbins)-1):
            inbin = (tform>=tbins[t]) & (tform<tbins[t+1])
            Ncl, _ = np.histogram(np.log10(mass[inbin]), bins=logmbins)
            dNcl = Ncl /np.diff(np.power(10,logmbins))
            #ax.stairs(np.log10(dNcl), logmbins, 
            #          lw=lw, color=colors[t],
            #          )
            ax.plot(midbins, np.log10(dNcl),
                    lw=lw, color=colors[t],
                    )

        ## ADD BEST FIT POWER LAW MODEL ##

    print('Total number of clusters:', ntot)
    #ax.fill_between(logmbins[logmbins<np.log10(6e4)], y1=1, y2=8,
    #                facecolor='gray', alpha=.5,
    #                )

    ## Plot reference power-law with slope -2 ##
    logm = np.logspace(3,8,1000)
    pl = np.power(logm, -2)
    pl *= ntot /10 /np.trapezoid(x=logm, y=pl)
    ax.plot(np.log10(logm), np.log10(pl), c='k', ls='--', lw=lw,
            label='Power-law with slope -2',
            )

    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=3, xmax=8)
    ax.set_xticks(np.linspace(3,8,6))
    ax.set_ylabel(r'$\mathrm{log}(\mathrm{dN}/\mathrm{d}M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    #ax.set_ylim(ymin=1, ymax=7.5)
    ax.set_box_aspect(1)
    #ax.legend(loc='upper right', fontsize=fs)
    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'CMF.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'

    plot_CMF()
    plt.show()
