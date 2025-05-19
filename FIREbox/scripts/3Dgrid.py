import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os

FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'


def plot_clusterGrid(Zsun=0.017):

    cluster_file = os.path.join(FIREbox_dir, 'gridClusters.hdf5')
    with h5py.File(cluster_file) as c:
        cluster_masses = c['Mass'][()]
        cluster_radii = c['HalfMassRadius'][()]
        cluster_Z = c['Metallicity'][()]
        cluster_ids = c['id'][()]
        cluster_nBBHs = c['nBBHs'][()]
    cluster_ids = np.astype(cluster_ids, np.str_)
    print('Number of clusters in the grid:', len(cluster_ids))
    print('Number of clusters having at least 1 BBH merger:', np.sum(cluster_nBBHs>0))
    print('Max number of BBH mergers in a single cluster:', np.max(cluster_nBBHs))

    """
    bbh_file = os.path.join(FIREbox_dir, 'allBBHs.dat')
    cluster_id = np.loadtxt(bbh_file, usecols=0, skiprows=1, dtype=np.str_).T
    unique, counts = np.unique(cluster_id, return_counts=True)
    print('Number of clusters having at least 1 BBH merger:', len(unique))
    print('Max number of BBH mergers in a single cluster:', np.max(counts))
    """

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.LogNorm(vmin=1e0, vmax=2e3)

    fig = plt.figure(figsize=(3.5,3.5))
    ax = fig.add_subplot(111, projection='3d')

    # Clusters with no BBH merger
    quiet = cluster_nBBHs==0
    ax.scatter(np.log10(cluster_masses[quiet]), 
               np.log10(cluster_radii[quiet]), 
               np.log10(cluster_Z[quiet] /Zsun),
               marker='o', c='gray', s=2**2, 
               zorder=0, depthshade=False,
               )
    # Clusters that produce at least 1 BBH merger
    ax.scatter(np.log10(cluster_masses[~quiet]),
               np.log10(cluster_radii[~quiet]),
               np.log10(cluster_Z[~quiet] /Zsun),
               marker='o', c=cmap(norm(cluster_nBBHs[~quiet])), s=3**2, 
               zorder=1, depthshade=False,
               )


    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=4.5, xmax=7.)
    ax.set_xticks(np.linspace(5.,6.5,4))
    ax.set_ylabel(r'$\mathrm{log}(r_\mathrm{hm}/\mathrm{pc})$', fontsize=fs)
    ax.set_ylim(ymin=-1.5, ymax=2.5)
    ax.set_yticks(np.linspace(-1,2,4))
    ax.set_zlabel(r'$\mathrm{log}(Z_\mathrm{cl}/\mathrm{Z}_\odot)$', fontsize=fs)
    ax.set_zlim(zmin=-2, zmax=.5)
    ax.set_zticks(np.linspace(-1.5,0.,4))

    ax.view_init(azim=-60, elev=15)

    # Save
    plot_dir = os.path.join(FIREbox_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'3DGrid.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    plot_clusterGrid()
    plt.show()
