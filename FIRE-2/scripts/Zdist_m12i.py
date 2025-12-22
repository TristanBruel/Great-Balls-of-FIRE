import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os

import gizmo_analysis as gizmo
import utilities as ut


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

    #simu = 'm12i'
    #sim_dir = os.path.join(FIRE_dir,simu+'_res7100')

    ## Use the m12i MHD version of GBoF I & II ##
    sim_dir = os.path.join(FIRE_dir,'m12i_MHD_res7100')
    ## All stars ##
    snap_dir = os.path.join(sim_dir,'output')
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0,
            simulation_directory=sim_dir,
            snapshot_directory=snap_dir,
            )
    h = part.Cosmology['hubble']
    t0 = part.snapshot['time']
    star_particles = part['star']
    masses = star_particles['mass']
    metallicities = part['star'].prop('massfraction.metals')

    Mstar, _ = np.histogram(np.log10(metallicities/Zsun), bins=logZbins, weights=masses)
    Mstar = Mstar /np.diff(logZbins) /np.sum(Mstar)
    ax.stairs(Mstar, logZbins,
              color='k', lw=lw, label='All stars',
              )


    ## All star clusters ##
    #cluster_file = os.path.join(sim_dir, 'AllClusters_m12i.hdf5')
    cluster_file = os.path.join(sim_dir, 'ClusterPopulation_1_reduced.hdf5')
    with h5py.File(cluster_file) as c:
        mass = c['Mass'][()]
        metal = c['Metallicity'][()]

    Mcl, _ = np.histogram(np.log10(metal/Zsun), bins=logZbins, weights=mass)
    Mcl = Mcl /np.diff(logZbins) /np.sum(Mcl)
    ax.stairs(Mcl, logZbins, 
              color=CB_color_cycle[0], lw=lw, label='All clusters',
              )

    ## Massive clusters ##
    massive = mass>6e4
    Mmassive, _ = np.histogram(np.log10(metal[massive]/Zsun), bins=logZbins, weights=mass[massive])
    Mmassive = Mmassive /np.diff(logZbins) /np.sum(Mmassive)
    ax.stairs(Mmassive, logZbins, 
              color=CB_color_cycle[0], lw=lw, fill=True, alpha=.4,
              label='Massive clusters',
            )

    ## Clusters that produce merging BBHs ##
    grid_cluster_file = os.path.join(FIRE_dir, 'grid/gridClusters.hdf5')
    with h5py.File(grid_cluster_file) as c:
        mass = c['Mass'][()]
        metal = c['Metallicity'][()]
        clid = c['id'][()]
    clid = np.array(clid, dtype=str)
    bbhfile = os.path.join(FIRE_dir, 'grid/allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    in_sim = np.array([c.split('_')[0]=='m12i' for c in cluster_id])
    unique, indexes = np.unique(cluster_id[in_sim], return_index=True)
    lochost = np.array([np.where(clid==i)[0][0] for i in unique])
    #lochost = np.array([int(u.split('_')[1]) for u in unique])
    Mhost, _ = np.histogram(np.log10(metal[lochost]/Zsun), bins=logZbins, weights=mass[lochost])
    Mhost = Mhost /np.diff(logZbins) /np.sum(Mhost)
    ax.stairs(Mhost, logZbins,
              color=CB_color_cycle[0], lw=lw, hatch='//',
              label='BBH-forming',
            )


    ax.set_xlabel(r'$\mathrm{log}_{10}(Z/\mathrm{Z}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=-4.2, xmax=1.2)
    ax.set_xticks(np.linspace(-4,1,6))
    ax.set_ylabel(r'$\mathrm{dn}/\mathrm{dlog}_{10}(Z/\mathrm{Z}_\odot)$', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-3, ymax=1e1)
    ax.set_box_aspect(1)
    ax.legend(loc='upper left', fontsize=fs)

    ax.text(0.3, 4, 'm12i', fontsize=fs,
            bbox=dict(boxstyle='round',facecolor='none',edgecolor='k'),
            )
    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'Zdist_m12i.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'

    plot_Zdist()
    plt.show()
