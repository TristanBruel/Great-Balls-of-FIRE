import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os


def plot_Mcl_rhm():

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

    logmbins = np.linspace(3, 7, 81)
    logrbins = np.linspace(-2, 3, 101)
    X, Y = np.meshgrid(logmbins[:-1],logrbins[:-1])
    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.LogNorm(vmin=1,vmax=6e2)

    for n,simu in enumerate(['m11i','m11q','m11e','m11h']):
        ax = axes[n]
        sim_dir = os.path.join(FIRE_dir,simu+'_res7100')

        ## All clusters ##
        cluster_file = os.path.join(sim_dir, 'AllClusters_%s.hdf5' %simu)
        with h5py.File(cluster_file) as c:
            mass = c['Mass'][()]
            radius = c['HalfMassRadius'][()]

        H, _, _ = np.histogram2d(x=np.log10(mass), y=np.log10(radius), bins=[logmbins,logrbins])
        ax.pcolormesh(X, Y, H.T,
                      cmap=cmap,
                      norm=norm,
                      )


        ## Clusters that produce merging BBHs ##
        grid_cluster_file = os.path.join(FIRE_dir, 'grid/gridClusters.hdf5')
        with h5py.File(grid_cluster_file) as c:
            mass = c['Mass'][()]
            radius = c['HalfMassRadius'][()]
            clid = c['id'][()]
            nbbhs = c['nBBHs'][()]
        clid = np.array(clid, dtype=str)
        in_sim = np.array([c.split('_')[0]==simu for c in clid])
        has_bbhs = nbbhs[in_sim]>0
        if np.sum(has_bbhs)>0:
            size = (2 + 3*np.log10(nbbhs[in_sim][has_bbhs]))**2
            sc = ax.scatter(np.log10(mass[in_sim][has_bbhs]),
                            np.log10(radius[in_sim][has_bbhs]),
                            c='r', s=size, edgecolor='white', lw=lw-.5,
                            )


        ## Print lines of constant density ##
        densities = np.array([1, 1e3, 1e6, 1e9])
        Mplot = np.logspace(-4,8,4001)
        for k,rho in enumerate(densities):
            rplot = (Mplot /(4/3 *np.pi *rho))**(1/3)
            ax.plot(np.log10(Mplot), np.log10(rplot),
                    c='dimgray', linestyle='--', zorder=0,
                    )


        ax.fill_between(logmbins[logmbins<np.log10(6e4)], y1=-2, y2=2.25,
                        facecolor='gray', alpha=.5, zorder=10,
                        )


        ax.set_xlim(xmin=4, xmax=7.25)
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.set_ylim(ymin=-2, ymax=2.25)
        ax.tick_params(axis='y', which='minor', left=False)
        ax.set_box_aspect(1)

        ax.text(6., -1.75, simu, fontsize=fs, 
                bbox=dict(boxstyle='round',facecolor='none',edgecolor='k'),
                )

    axes[0].set_xticks([])
    axes[0].set_ylabel(r'$r_\mathrm{hm}\ [\mathrm{pc}]$', fontsize=fs)
    axes[0].set_yticks(np.linspace(-1,2,4),[r'$0.1$',r'$1$',r'$10$',r'$100$'])

    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].set_xlabel(r'$M_\mathrm{cl}\ [\mathrm{M}_\odot]$', fontsize=fs)
    axes[2].set_xticks(np.linspace(3,7,3),[r'$10^3$',r'$10^5$',r'$10^7$'])
    axes[2].set_ylabel(r'$R_\mathrm{GMC}\ [\mathrm{pc}]$', fontsize=fs)
    axes[2].set_yticks(np.linspace(-2,2,5),[r'$0.01$',r'$0.1$',r'$1$',r'$10$',r'$100$'])

    axes[3].set_xlabel(r'$M_\mathrm{GMC}\ [\mathrm{M}_\odot]$', fontsize=fs)
    axes[3].set_xticks(np.linspace(5,7,2),[r'$10^5$',r'$10^7$'])
    axes[3].set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.086, pad=0.04)#, aspect=10)
    cbar.set_label(r'Number of clusters', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
    cbar.set_ticks([1,10,100])
    cbar.set_ticklabels([r'$1$',r'$10$',r'$10^2$'])
        
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'clusters_4gal.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    plot_Mcl_rhm()
    plt.show()
