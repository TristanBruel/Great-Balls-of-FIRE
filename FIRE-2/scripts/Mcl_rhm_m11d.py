import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import argparse


def plot_Mcl_rhm(add_leg=True, add_bar=False):

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

    logmbins = np.linspace(3, 7, 81)
    logrbins = np.linspace(-2, 3, 101)
    X, Y = np.meshgrid(logmbins[:-1],logrbins[:-1])
    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.LogNorm(vmin=1,vmax=6e2)

    sim_dir = os.path.join(FIRE_dir,'m11d_res7100')

    ## All star clusters ##
    cluster_file = os.path.join(sim_dir, 'AllClusters_m11d.hdf5')
    with h5py.File(cluster_file) as c:
        mass = c['Mass'][()]
        radius = c['HalfMassRadius'][()]

    tokeep = mass>=1e4
    mass = mass[tokeep]
    radius = radius[tokeep]
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
    in_sim = np.array([c.split('_')[0]=='m11d' for c in clid])
    has_bbhs = nbbhs[in_sim]>0
    if np.sum(has_bbhs)>0:
        size = (2 + 3*np.log10(nbbhs[in_sim][has_bbhs]))**2
        sc = ax.scatter(np.log10(mass[in_sim][has_bbhs]), 
                        np.log10(radius[in_sim][has_bbhs]), 
                        c='r', s=size, edgecolor='white', lw=lw-.5,
                        )

    if add_leg:
        kw = dict(prop="sizes", num=[1,10,100,1000], color='r', fmt=r"${x:.0f}$", func=lambda s: 10**((np.sqrt(s)-2)/3))
        ax.legend(*sc.legend_elements(**kw), loc="lower right", title="BBH mergers")


    ## Print lines of constant density ##
    densities = np.array([1, 1e3, 1e6, 1e9])
    Mplot = np.logspace(-4,8,4001)
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


    ax.fill_between(logmbins[logmbins<np.log10(6e4)], y1=-2, y2=2.25,
                    facecolor='gray', alpha=.5, zorder=10,
                    )


    ax.set_xlabel(r'$M_\mathrm{cl}\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(xmin=4, xmax=7.25)
    ax.set_xticks(np.linspace(4,7,4),[r'$10^4$',r'$10^5$',r'$10^6$',r'$10^7$'])
    ax.set_ylabel(r'$r_\mathrm{hm}\ [\mathrm{pc}]$', fontsize=fs)
    ax.set_ylim(ymin=-2, ymax=2.25)
    ax.set_yticks(np.linspace(-2,2,5),[r'$0.01$',r'$0.1$',r'$1$',r'$10$',r'$100$'])
    ax.set_box_aspect(1)

    ax.text(6.5, 1.75, 'm11d', fontsize=fs, 
            bbox=dict(boxstyle='round',facecolor='none',edgecolor='k'),
            )

    if add_bar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cbar.set_label(r'Number of clusters', fontsize=fs)
        cbar.ax.tick_params(labelsize=0.8*fs)
        cbar.set_ticks([1,10,100])
        cbar.set_ticklabels([r'$1$',r'$10$',r'$10^2$'])
    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'clusters_m11d.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--leg', dest='leg', action='store_true', help='Whether to add legend to the plot', default=False)
    parser.add_argument('--bar', dest='bar', action='store_true', help='Whether to add colorbar to the plot', default=False)
    args = parser.parse_args()

    plot_Mcl_rhm(add_leg=args.leg, add_bar=args.bar)
    plt.show()
