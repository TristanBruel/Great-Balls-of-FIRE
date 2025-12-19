import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os


def plot_clusterGrid(Zsun=0.017):

    cluster_file = os.path.join(grid_dir, 'gridClusters_withKremer.hdf5')
    with h5py.File(cluster_file) as c:
        cluster_masses = c['Mass'][()]
        cluster_radii = c['HalfMassRadius'][()]
        cluster_Z = c['Metallicity'][()]
        cluster_nBBHs = c['nBBHs'][()]
        cluster_id = c['id'][()]
    cluster_id = np.array(cluster_id, dtype='str')
    isKremer = np.array([f.split('_')[0]=='Kremer' for f in cluster_id])
    print('Number of clusters in the grid:', len(cluster_Z))
    print('Number of clusters from Kyle Kremer simulations:', np.sum(isKremer))
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

    fig = plt.figure(figsize=(4.5,4.5))
    ax = fig.add_subplot(111, projection='3d')

    # Clusters with no BBH merger
    quiet = cluster_nBBHs==0
    ax.scatter(np.log10(cluster_masses[quiet]), 
               np.log10(cluster_radii[quiet]), 
               np.log10(cluster_Z[quiet] /Zsun),
               marker='o', c='gray', s=2**2, 
               depthshade=False,
               )
    # Clusters that produce at least 1 BBH merger
    ax.scatter(np.log10(cluster_masses[~quiet]),
               np.log10(cluster_radii[~quiet]),
               np.log10(cluster_Z[~quiet] /Zsun),
               marker='o', c=cmap(norm(cluster_nBBHs[~quiet])), s=3**2, 
               depthshade=False,
               )

    # Clusters from Kyle Kremer simulations
    ax.scatter(np.log10(cluster_masses[isKremer]),
               np.log10(cluster_radii[isKremer]),
               np.log10(cluster_Z[isKremer] /Zsun),
               marker='o', c='r', s=5**2,
               depthshade=False,
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
    #ax.set_box_aspect(None, zoom=0.75)
    #plt.subplots_adjust(left=0.01,right=.8,
    #                    bottom=0.01, top=1.3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.2, aspect=10)
    cbar.set_label(r'Number of merging BBHs', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)

    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'3Dgrid.pdf'), format='pdf', 
                bbox_inches='tight',
                )


##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
    grid_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2/grid'

    plot_clusterGrid()
    plt.show()
