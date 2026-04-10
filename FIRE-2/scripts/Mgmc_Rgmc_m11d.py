import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import argparse


def plot_Mgmc_Rgmc(add_leg=False, add_bar=False):

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

    logmbins = np.linspace(4, 9, 101)
    logrbins = np.linspace(0, 4, 81)
    X, Y = np.meshgrid(logmbins[:-1],logrbins[:-1])
    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.LogNorm(vmin=1,vmax=1e3)

    sim_dir = os.path.join(FIRE_dir,'m11d_res7100')

    ## All clouds ##
    cloud_file = os.path.join(sim_dir, 'AllGMCs_m11d.hdf5')
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
    cluster_file = os.path.join(sim_dir, 'AllMassiveClusters_m11d.hdf5')
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

    if add_leg:
        kw = dict(prop="sizes", num=[1,10,100], color='r', fmt=r"${x:.0f}$", func=lambda s: 10**((np.sqrt(s)-2)/3))
        ax.legend(*sc.legend_elements(**kw), loc="lower right", title="Massive clusters")


    ## Print lines of constant surface density ##
    densities = np.array([1, 10, 1e2, 1e3])
    Mplot = np.logspace(4,9,5001)
    labels = np.array([r'$%i$' %rho for rho in densities[1:]])
    labels = np.append(r'$\Sigma_\mathrm{GMC}= 1\ \mathrm{M}_\odot \mathrm{pc}^{-2}$',labels)
    xtext = np.repeat(4.1, 4)
    ytext = np.log10((np.power(10,xtext) /(np.pi*densities))**(1/2)) +0.05
    ytext[-1] += 0.05
    for k,rho in enumerate(densities):
        rplot = (Mplot /(np.pi*rho))**(1/2)
        ax.plot(np.log10(Mplot), np.log10(rplot),
                c='dimgray', linestyle='--', zorder=0,
                )
        ax.text(xtext[k], ytext[k], labels[k],
                c='dimgray', fontsize=fs,
                rotation=33, rotation_mode='anchor',
                )



    ax.set_xlabel(r'$M_\mathrm{GMC}\ [\mathrm{M}_\odot]$', fontsize=fs)
    #ax.set_xscale('log')
    ax.set_xlim(xmin=4, xmax=8.5)
    ax.set_xticks(np.linspace(4,8,5),[r'$10^4$',r'$10^5$',r'$10^6$',r'$10^7$',r'$10^8$'])
    ax.set_ylabel(r'$R_\mathrm{GMC}\ [\mathrm{pc}]$', fontsize=fs)
    #ax.set_yscale('log')
    ax.set_ylim(ymin=0, ymax=3.25)
    ax.set_yticks(np.linspace(0,3,4),[r'$1$',r'$10$',r'$10^2$',r'$10^3$'])
    ax.set_box_aspect(1)

    ax.text(4.25, 3., 'm11d', fontsize=fs, 
            bbox=dict(boxstyle='round',facecolor='none',edgecolor='k'),
            )

    if add_bar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=20)
        cbar.set_label(r'Number of GMCs', fontsize=fs)
        cbar.ax.tick_params(labelsize=0.8*fs)
        cbar.set_ticks([1,10,100,1000])
        cbar.set_ticklabels([r'$1$',r'$10$',r'$10^2$',r'$10^3$'])
    
    # Save
    plot_dir = os.path.join(FIRE_dir, 'plots')
    plt.savefig(os.path.join(plot_dir,'clouds_m11d.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--leg', dest='leg', action='store_true', help='Whether to add legend to the plot', default=False)
    parser.add_argument('--bar', dest='bar', action='store_true', help='Whether to add colorbar to the plot', default=False)
    args = parser.parse_args()

    plot_Mgmc_Rgmc(add_leg=args.leg, add_bar=args.bar)
    plt.show()
