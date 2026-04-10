import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os


def plot_Mgmc_Rgmc():

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#e41a1c']

    fig, axes = plt.subplots(2, 2, figsize=(3.97,3.5))
    # Remove vertical and horizontal space between Axes
    fig.subplots_adjust(hspace=0, wspace=0)
    axes = axes.flatten()

    logmbins = np.linspace(4, 9, 101)
    logrbins = np.linspace(0, 4, 81)
    X, Y = np.meshgrid(logmbins[:-1],logrbins[:-1])
    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.LogNorm(vmin=1,vmax=1e3)

    for n,simu in enumerate(['m11i','m11q','m11e','m11h']):
        ax = axes[n]
        sim_dir = os.path.join(FIRE_dir,simu+'_res7100')

        ## All clouds ##
        cloud_file = os.path.join(sim_dir, 'AllGMCs_%s.hdf5' %simu)
        with h5py.File(cloud_file) as c:
            mass = c['Mass'][()]
            #radius = c['HalfMassRadius'][()]
            radius = c['Reff'][()]
            snap = c['Snapshot'][()]

        H, _, _ = np.histogram2d(x=np.log10(mass), y=np.log10(radius), bins=[logmbins,logrbins])
        ax.pcolormesh(X, Y, H.T,
                      cmap=cmap,
                      norm=norm,
                      )


        ## GMCs that produce massive clusters #
        cluster_file = os.path.join(sim_dir, 'AllMassiveClusters_%s.hdf5' %simu)
        with h5py.File(cluster_file) as c:
            cluster_mass = c['Mass'][()]
            cluster_radius = c['HalfMassRadius'][()]
            GMC_id = c['GMC_ID'][()]
            snap_cluster = c['Snapshot'][()]
        host_id = np.zeros(len(cluster_mass),dtype='U10')
        host_mass = np.zeros(len(cluster_mass))
        host_radius = np.zeros(len(cluster_mass))
        for n in range(len(cluster_mass)):
            host_id[n] = 'snap%i_%i' %(snap_cluster[n],GMC_id[n])
            host_mass[n] = mass[snap==snap_cluster[n]][GMC_id[n]]
            host_radius[n] = radius[snap==snap_cluster[n]][GMC_id[n]]
        unique, indices, n_clusters = np.unique(host_id, return_index=True, return_counts=True)
        size = (2 + 3*np.log10(n_clusters))**2
        sc = ax.scatter(np.log10(host_mass[indices]), np.log10(host_radius[indices]),
                        c='r', s=size, edgecolor='white', lw=lw-.5,
                        )

        ## Print lines of constant surface density ##
        densities = np.array([1, 10, 1e2, 1e3])
        Mplot = np.logspace(4,9,5001)
        for k,rho in enumerate(densities):
            rplot = (Mplot /(np.pi*rho))**(1/2)
            ax.plot(np.log10(Mplot), np.log10(rplot),
                    c='dimgray', linestyle='--', zorder=0,
                    )


        ax.set_xlim(xmin=4, xmax=8.5)
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.set_ylim(ymin=0, ymax=3.25)
        ax.tick_params(axis='y', which='minor', left=False)
        ax.set_box_aspect(1)

        ax.text(7., 0.25, simu, fontsize=fs, 
                bbox=dict(boxstyle='round',facecolor='none',edgecolor='k'),
                )

    axes[0].set_xticks([])
    axes[0].set_ylabel(r'$R_\mathrm{GMC}\ [\mathrm{pc}]$', fontsize=fs)
    axes[0].set_yticks(np.linspace(1,3,3),[r'$10$',r'$10^2$',r'$10^3$'])

    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].set_xlabel(r'$M_\mathrm{GMC}\ [\mathrm{M}_\odot]$', fontsize=fs)
    axes[2].set_xticks(np.linspace(4,8,3),[r'$10^4$',r'$10^6$',r'$10^8$'])
    axes[2].set_ylabel(r'$R_\mathrm{GMC}\ [\mathrm{pc}]$', fontsize=fs)
    axes[2].set_yticks(np.linspace(0,3,4),[r'$1$',r'$10$',r'$10^2$',r'$10^3$'])

    axes[3].set_xlabel(r'$M_\mathrm{GMC}\ [\mathrm{M}_\odot]$', fontsize=fs)
    axes[3].set_xticks(np.linspace(6,8,2),[r'$10^6$',r'$10^8$'])
    axes[3].set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.086, pad=0.04)#, aspect=10)
    cbar.set_label(r'Number of GMCs', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
    cbar.set_ticks([1,10,100,1000])
    cbar.set_ticklabels([r'$1$',r'$10$',r'$10^2$',r'$10^3$'])
        
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'clouds_4gal.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    plot_Mgmc_Rgmc()
    plt.show()
