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

    fig, axes = plt.subplots(1, 2, figsize=(8.,2.75))

    ################
    ## Mcl vs rhm ##
    ################
    ax = axes[0]

    # Define metallicity range of interest
    #Zrange = np.array([5e-2,5e-1]) *Zsun
    #inrange = (cluster_Z>=Zrange[0]) & (cluster_Z<Zrange[1])
    #print('Number of clusters in metallicity range [{:.1e},{:.1e}]:'.format(Zrange[0],Zrange[1]))
    #print(np.sum(inrange))
    #cluster_masses = cluster_masses[inrange]
    #cluster_radii = cluster_radii[inrange]
    #cluster_nBBHs = cluster_nBBHs[inrange]

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
    Mplot = np.logspace(-4,7,3001)
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

    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=4.5, xmax=7.)
    ax.set_xticks(np.linspace(5.,6.5,4))
    ax.set_ylabel(r'$\mathrm{log}(r_\mathrm{hm}/\mathrm{pc})$', fontsize=fs)
    ax.set_ylim(ymin=-1.7, ymax=2.3)
    ax.set_yticks(np.linspace(-1,2,4))
    ax.set_box_aspect(1)

    ################
    ## Mcl vs Zcl ##
    ################
    ax = axes[1]

    # Clusters with no BBH merger
    ax.scatter(np.log10(cluster_masses[quiet]),
               np.log10(cluster_Z[quiet]/Zsun),
               c='gray', s=2**2,
               )
    # Clusters that produce at least 1 BBH merger
    ax.scatter(np.log10(cluster_masses[~quiet]),
               np.log10(cluster_Z[~quiet]/Zsun),
               c=cmap(norm(cluster_nBBHs[~quiet])), s=3**2,
               )


    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$', fontsize=fs)
    ax.set_xlim(xmin=4.5, xmax=7.)
    ax.set_xticks(np.linspace(5.,6.5,4))
    ax.set_ylabel(r'$\mathrm{log}(Z_\mathrm{cl}/\mathrm{Z}_\odot)$', fontsize=fs)
    ax.set_ylim(ymin=-2, ymax=.5)
    ax.set_yticks(np.linspace(-1.5,0.,4))
    ax.set_box_aspect(1)

    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #cbar = fig.colorbar(sm, ax=axes, fraction=0.086, pad=0.04, aspect=10)
    #cbar.set_label(r'Number of merging BBHs', fontsize=fs)
    #cbar.ax.tick_params(labelsize=0.8*fs)
    
    # Save
    plot_dir = os.path.join(FIREbox_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'2Dgrid.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
    grid_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2/grid'

    plot_clusterGrid()
    plt.show()
