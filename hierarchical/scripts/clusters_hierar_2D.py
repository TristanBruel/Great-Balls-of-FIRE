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
        cluster_metallicities = c['Metallicity'][()]
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

    fig, axes = plt.subplots(1, 2, figsize=(9.,3.))

    ## First plot: Mcl vs rhm ##
    ax=axes[0]
    # Clusters with no BBH merger
    quiet = cluster_nBBHs==0
    ax.scatter(np.log10(cluster_masses[quiet]),
               np.log10(cluster_radii[quiet]),
               c='gray', s=2**2,
               )
    # Clusters that produce at least 1 BBH merger
    ax.scatter(np.log10(cluster_masses[~quiet]),
               np.log10(cluster_radii[~quiet]),
               c=cmap(norm(cluster_nBBHs[~quiet])), s=3**2,
               )

    # Print lines of constant density
    densities = np.array([1, 1e3, 1e6, 1e9])
    Mplot = np.logspace(4.5,8,1001)
    labels = np.array([r'$10^{%i}$' %np.log10(rho) for rho in densities[:-1]])
    labels = np.append(labels, r'$\rho_\mathrm{cl}= 10^9\ \mathrm{M}_\odot \mathrm{pc}^{-3}$')
    xtext = np.array([5.8, 6.7, 6.7, 5.3])
    ytext = np.log10((np.power(10,xtext) /(4/3 *np.pi *densities))**(1/3)) +0.05
    ytext[-1] += 0.05
    for k,rho in enumerate(densities):
        rplot = (Mplot /(4/3 *np.pi *rho))**(1/3)
        ax.plot(np.log10(Mplot), np.log10(rplot),
                c='dimgray', linestyle='--', zorder=0,
                )
        ax.text(xtext[k], ytext[k], labels[k], 
                c='dimgray', fontsize=fs,
                rotation=12, rotation_mode='anchor',
                )

    ### Show clusters that host hierarchical BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    mBH1, mBH2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    m1 = np.maximum(mBH1, mBH2)
    hierar = a1+a2>0
    unique, indexes = np.unique(cluster_id[hierar], return_index=True)
    lochost = np.array([np.where(cluster_ids==u)[0][0] for u in unique])
    ax.scatter(np.log10(cluster_masses[lochost]),
               np.log10(cluster_radii[lochost]),
               c='r', s=3**2,
               )

    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=4.5, xmax=7.1)
    ax.set_xticks(np.linspace(5.,7.,5))
    ax.set_ylabel(r'$\mathrm{log}(r_\mathrm{hm}/\mathrm{pc})$', fontsize=fs)
    #ax.set_ylim(ymin=-1.7, ymax=2.3)
    ax.set_yticks(np.linspace(-1,2,4))
    ax.set_box_aspect(1)


    ## Second plot: Mcl vs Z ##
    ax=axes[1]
    # Clusters with no BBH merger
    ax.scatter(np.log10(cluster_masses[quiet]),
               np.log10(cluster_metallicities[quiet] /Zsun),
               c='gray', s=2**2,
               )
    # Clusters that produce at least 1 BBH merger
    ax.scatter(np.log10(cluster_masses[~quiet]),
               np.log10(cluster_metallicities[~quiet] /Zsun),
               c=cmap(norm(cluster_nBBHs[~quiet])), s=3**2,
               )

    ### Show clusters that host massive BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    mBH1, mBH2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    m1 = np.maximum(mBH1, mBH2)
    hierar = (a1+a2)>0
    print('Number of hierarchical BBH mergers:', np.sum(hierar))
    unique, indexes = np.unique(cluster_id[hierar], return_index=True)
    print('Number of clusters hosting hierarchical mergers:', len(unique))
    lochost = np.array([np.where(cluster_ids==u)[0][0] for u in unique])
    ax.scatter(np.log10(cluster_masses[lochost]),
               np.log10(cluster_metallicities[lochost] /Zsun),
               c='r', s=3**2, label=r'hierarchical BBH mergers',
               )

    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=4.5, xmax=7.1)
    ax.set_xticks(np.linspace(5.,7.,5))
    ax.set_ylabel(r'$\mathrm{log}(Z_\mathrm{cl}/\mathrm{Z}_\odot)$', fontsize=fs)
    #ax.set_ylim(ymin=-2, ymax=.5)
    ax.set_yticks(np.linspace(-2.,0.5,6))
    ax.legend(loc='lower right', fontsize=fs)
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.086, pad=0.06, aspect=10)
    cbar.set_label(r'Number of merging BBHs', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
    
    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'2DGrid_hierar.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
    grid_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2/grid'

    plot_clusterGrid()
    plt.show()
