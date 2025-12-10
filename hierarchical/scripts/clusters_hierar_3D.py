import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os


def plot_clusterGrid(Zsun=0.017):

    cluster_file = os.path.join(grid_dir, 'gridClusters.hdf5')
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


    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.LogNorm(vmin=1e0, vmax=2e3)

    fig = plt.figure(figsize=(6.5,6.5))
    ax = fig.add_subplot(111, projection='3d')

    # Clusters with no BBH merger
    quiet = cluster_nBBHs==0
    ax.scatter(np.log10(cluster_masses[quiet]),
               np.log10(cluster_radii[quiet]),
               np.log10(cluster_Z[quiet] /Zsun),
               c='gray', s=2**2,
               )
    # Clusters that produce at least 1 BBH merger
    ax.scatter(np.log10(cluster_masses[~quiet]),
               np.log10(cluster_radii[~quiet]),
               np.log10(cluster_Z[~quiet] /Zsun),
               c=cmap(norm(cluster_nBBHs[~quiet])), s=3**2,
               )

    ### Show clusters that host hierarchical BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
    hierar = (a1+a2)>0
    unique, indexes = np.unique(cluster_id[hierar], return_index=True)
    lochost = np.array([np.where(cluster_ids==u)[0][0] for u in unique])
    ax.scatter(np.log10(cluster_masses[lochost]),
               np.log10(cluster_radii[lochost]),
               np.log10(cluster_Z[lochost] /Zsun),
               c='r', s=5**2,
               )

    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=4.5, xmax=7.1)
    ax.set_xticks(np.linspace(5.,7.,5))
    ax.set_ylabel(r'$\mathrm{log}(r_\mathrm{hm}/\mathrm{pc})$', fontsize=fs)
    ax.set_ylim(ymin=-1.5, ymax=2.5)
    ax.set_yticks(np.linspace(-1,2,4))
    ax.set_zlabel(r'$\mathrm{log}(Z_\mathrm{cl}/\mathrm{Z}_\odot)$', fontsize=fs)
    ax.set_zlim(zmin=-2, zmax=.5)
    ax.set_zticks(np.linspace(-1.5,0.,4))
    #ax.set_box_aspect(1)


    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.2, aspect=10)
    cbar.set_label(r'Number of merging BBHs', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
    
    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'3DGrid_hierar.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
    grid_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2/grid'

    plot_clusterGrid()
    plt.show()
