import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os



def plot_clusterGrid(Zsun=0.017):

    cluster_file = 'sample_m12i_LHS.h5'
    with h5py.File(cluster_file) as c:
        cluster_masses = c['Mass'][()]
        cluster_radii = c['HalfMassRadius'][()]
        cluster_Z = c['Metallicity'][()]
        cluster_R = c['Rgal'][()]
    print('Number of clusters in the grid:', len(cluster_masses))

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    #cmap = plt.get_cmap('viridis')
    #norm = mpl.colors.LogNorm(vmin=1e0, vmax=2e3)

    fig = plt.figure(figsize=(6.5,6.5))
    ax = fig.add_subplot(111, projection='3d')

    # Clusters with no BBH merger
    ax.scatter(np.log10(cluster_masses), 
               np.log10(cluster_radii), 
               np.log10(cluster_Z /Zsun),
               marker='o', c='gray', s=2**2, 
               zorder=0, depthshade=False,
               )

    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    #ax.set_xlim(xmin=3, xmax=5)
    ax.set_xticks(np.linspace(3,5,5))
    ax.set_ylabel(r'$\mathrm{log}(r_\mathrm{hm}/\mathrm{pc})$', fontsize=fs)
    #ax.set_ylim(ymin=-1.5, ymax=2.5)
    ax.set_yticks(np.linspace(-1,2,4))
    ax.set_zlabel(r'$\mathrm{log}(Z_\mathrm{cl}/\mathrm{Z}_\odot)$', fontsize=fs)
    #ax.set_zlim(zmin=-2, zmax=.5)
    ax.set_zticks(np.linspace(-1.5,0.,4))

    ax.view_init(azim=-60, elev=15)

    # Save
    plt.savefig('./grid_sample_LHS.png', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    plot_clusterGrid()
    plt.show()
