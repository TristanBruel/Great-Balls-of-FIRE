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
        cluster_metallicities = c['Metallicity'][()]
        cluster_ids = c['id'][()]
        cluster_nBBHs = c['nBBHs'][()]
    cluster_ids = np.astype(cluster_ids, np.str_)
    isKremer = np.array([f.split('_')[0]=='Kremer' for f in cluster_ids])
    print('Number of clusters in the grid:', len(cluster_ids))
    print('Number of clusters from Kyle Kremer simulations:', np.sum(isKremer))
    print('Number of clusters having at least 1 BBH merger:', np.sum(cluster_nBBHs>0))
    print('Max number of BBH mergers in a single cluster:', np.max(cluster_nBBHs))

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.,3.))
    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.LogNorm(vmin=1e0, vmax=2e3)

    mbins = np.linspace(4,8,41)
    rbins = np.linspace(-2,2,41)
    Zbins = np.linspace(-2,1,31)

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

    # Clusters from Kyle Kremer simulations
    #ax.scatter(np.log10(cluster_masses[isKremer]),
    #           np.log10(cluster_radii[isKremer]),
    #           c='r', s=4**2,
    #           )
    """
    H, xedges, yedges = np.histogram2d(x=np.log10(cluster_masses), 
                                       y=np.log10(cluster_radii), 
                                       bins=[mbins,rbins],
                                       density=True,
                                       )
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    H[H==0] = 1e-32
    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),10),
                     cmap=cmap,
                     norm=norm,
                    )
    """

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

    ### Show clusters that host GW241011-like BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs_withKremer.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1
    print('Total number of BBH mergers in the combined CMC catalog: {}'.format(len(m1)))
    print('Number of hierarchical BBH mergers: {}'.format(np.sum(a1+a2>0)))
    
    ## BBH mergers compatible with GW241011 and GW241110
    """
    # GW241110 posterior #
    gw241110_qm = np.loadtxt(os.path.join(w_dir,'gw241110_qm.txt'))
    q_bins_241110 = gw241110_qm[1:,0]
    log_m1_bins_241110 = gw241110_qm[0,1:]
    heights_241110_q = gw241110_qm[1:,1:]
    c90_241110_q = [gw241110_qm[0,0]]
    gw241110_am = np.loadtxt(os.path.join(w_dir,'gw241110_am.txt'))
    s1_bins_241110 = gw241110_am[1:,0]
    heights_241110_a = gw241110_am[1:,1:]
    c90_241110_a = [gw241110_am[0,0]]

    # GW241011-like BBH mergers #
    fig2, ax2 = plt.subplots(1, 1, figsize=(3.,3.))
    contour_q = ax2.contour(log_m1_bins_241110, q_bins_241110, heights_241110_q, 
                           levels=[*c90_241110_q, np.inf], colors=gw241110_color)
    p = contour_q.get_paths()[0]
    mask_q = p.contains_points(np.hstack((np.log10(m1)[:,np.newaxis],q[:,np.newaxis])))
    contour_a = ax2.contour(log_m1_bins_241110, s1_bins_241110, heights_241110_a, 
                           levels=[*c90_241110_a, np.inf], colors=gw241110_color)
    p = contour_a.get_paths()[0]
    mask_a = p.contains_points(np.hstack((np.log10(m1)[:,np.newaxis],a1[:,np.newaxis])))
    GW241110 = mask_q & mask_a
    plt.close(fig2)
    print('Number of GW241110-like BBH mergers:', np.sum(GW241110))
    """
    # BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses
    GW24x = (m1 < 30) & (q < 0.85) & (a1 > 0.05)
    unique, indexes = np.unique(cluster_id[GW24x], return_index=True)
    lochost = np.array([np.where(cluster_ids==u)[0][0] for u in unique])
    print('\n')
    print('Looking for BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses')
    print('Number of events found:', np.sum(GW24x))
    print('Number of clusters hosting these events:', len(lochost))
    print('Total number of BBH mergers from these clusters:',np.sum(cluster_nBBHs[lochost]),'\n')
    ax.scatter(np.log10(cluster_masses[lochost]),
               np.log10(cluster_radii[lochost]),
               edgecolor='r', facecolor=cmap(norm(cluster_nBBHs[lochost])), 
               linewidth=.6, s=5**2,
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

    # Clusters from Kyle Kremer simulations
    #ax.scatter(np.log10(cluster_masses[isKremer]),
    #           np.log10(cluster_metallicities[isKremer] /Zsun),
    #           c='r', s=4**2,
    #           )
    """
    H, xedges, yedges = np.histogram2d(x=np.log10(cluster_masses),
                                       y=np.log10(cluster_metallicities/Zsun),
                                       bins=[mbins,Zbins],
                                       density=True,
                                       )
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    H[H==0] = 1e-32
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),10),
                     cmap=cmap,
                     norm=norm,
                    )
    """

    ### Show clusters that host GW241011-like BBH mergers ###
    ax.scatter(np.log10(cluster_masses[lochost]),
               np.log10(cluster_metallicities[lochost] /Zsun),
               edgecolor='r', facecolor=cmap(norm(cluster_nBBHs[lochost])),
               linewidth=.6, s=5**2,
               label='hosts of GW24x\n-like mergers',
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
    #cbar.set_label(r'$\mathrm{d}^2\mathrm{N}_\mathrm{cl}/\mathrm{d}M_\mathrm{cl}\mathrm{d}X_\mathrm{cl}$', fontsize=fs)
    cbar.set_label(r'Number of BBH mergers', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
    
    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'2DGrid_hosts_GW24x.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    grid_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2/grid'

    GW24_dir = '/home/tristan-bruel/Documents/Science/o4a-astro/GW241011_GW241110_data_release/data'
    w_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical'
    gw241011_color = '#1f78b4'
    gw241110_color = '#33a02c'

    plot_clusterGrid()
    plt.show()
