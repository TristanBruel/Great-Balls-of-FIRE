import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

import os


def plot_mBH_spin(Zsun=0.017, n=1000):

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    fig, axes = plt.subplots(2, 2, figsize=(8,5))

    abins = np.linspace(0.,1.0,26)
    logm1bins = np.linspace(0.1,2.1,21)
    qbins = np.linspace(0.1,1.,10)


    ## Solar metallicities ##
    bbh_file = os.path.join(GW24_dir, 'external_cluster_simulations/cBHBd/Z_0.02.dat')
    m1, m2, a1, a2 = np.loadtxt(bbh_file, usecols=(2,3,14,15)).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1

    ### First-generation mergers ###
    m1g = m1[a1+a2==0]
    q1g = q[a1+a2==0]
    a1g = a1[a1+a2==0]
    ### Hierarchical mergers ###
    mng = m1[a1+a2>0]
    qng = q[a1+a2>0]
    ang = a1[a1+a2>0]

    print('Number of 1g+1g mergers in solar metallicity cBHB clusters:', len(m1g))

    # Spins #
    ax = axes[0,1]

    H, xedges, yedges = np.histogram2d(x=np.log10(mng), y=ang, bins=[logm1bins,abins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    H[H==0] = 1e-32
    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),10),
                     cmap=cmap,
                     norm=norm,
                    )

    ## Add GW241011 and GW241110 posteriors ##
    gw241011_am = np.loadtxt(os.path.join(w_dir,'gw241011_am.txt'))
    s1_bins_241011 = gw241011_am[1:,0]
    log_m1_bins_241011 = gw241011_am[0,1:]
    heights_241011 = gw241011_am[1:,1:]
    c90_241011 = [gw241011_am[0,0]]

    gw241110_am = np.loadtxt(os.path.join(w_dir,'gw241110_am.txt'))
    s1_bins_241110 = gw241110_am[1:,0]
    log_m1_bins_241110 = gw241110_am[0,1:]
    heights_241110 = gw241110_am[1:,1:]
    c90_241110 = [gw241110_am[0,0]]

    ax.contourf(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
    ax.contour(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10)
    patch1 = Patch(color=gw241110_color, alpha=0.5)

    ax.contourf(log_m1_bins_241011, s1_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
    ax.contour(log_m1_bins_241011, s1_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11)
    patch2 = Patch(color=gw241011_color, alpha=0.5)

    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([])
    #ax.set_ylabel(r'$\chi_1$', fontsize=fs)
    ax.set_ylim(ymin=0., ymax=1.0)
    ax.set_yticks(np.linspace(0.,1.0,6))
    ax.set_yticklabels([])
    #ax.set_box_aspect(1)
    ax.legend(handles=(patch1,patch2), labels=('GW241110','GW241011'),
              loc='lower right', fontsize=fs,
              )
    ax.grid(color='gray', linestyle='--')
    ax.set_title(r'$Z\sim\ \mathrm{Z}_\odot$', fontsize=fs)

    # Masses #
    ax = axes[1,1]

    H, xedges, yedges = np.histogram2d(x=np.log10(m1g), y=q1g, bins=[logm1bins,qbins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    H /= H.sum()
    t = np.linspace(0, H.max(), n)
    integral = ((H >= t[:, None, None]) * H).sum(axis=(1,2))
    contours = interp1d(integral, t)([0.95])
    ax.contour(X, Y, H.T, contours,
               colors=CB_color_cycle[3],
               linestyles='--',
               )

    print('Number of hierarchical mergers in solar metallicity cBHB clusters:', len(mng), '\n')
    H, xedges, yedges = np.histogram2d(x=np.log10(mng), y=qng, bins=[logm1bins,qbins])
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

    ## Add GW241011 and GW241110 posteriors ##
    gw241011_qm = np.loadtxt(os.path.join(w_dir,'gw241011_qm.txt'))
    q_bins_241011 = gw241011_qm[1:,0]
    log_m1_bins_241011 = gw241011_qm[0,1:]
    heights_241011 = gw241011_qm[1:,1:]
    c90_241011 = [gw241011_qm[0,0]]

    gw241110_qm = np.loadtxt(os.path.join(w_dir,'gw241110_qm.txt'))
    q_bins_241110 = gw241110_qm[1:,0]
    log_m1_bins_241110 = gw241110_qm[0,1:]
    heights_241110 = gw241110_qm[1:,1:]
    c90_241110 = [gw241110_qm[0,0]]

    ax.contourf(log_m1_bins_241110, q_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
    ax.contour(log_m1_bins_241110, q_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10)

    ax.contourf(log_m1_bins_241011, q_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
    ax.contour(log_m1_bins_241011, q_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11)

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([10, 30, 100])
    #ax.set_ylabel(r'$q$', fontsize=fs)
    ax.set_ylim(ymin=0.1, ymax=1.0)
    ax.set_yticks(np.linspace(0.2,1.0,5))
    ax.set_yticklabels([])
    #ax.set_box_aspect(1)
    ax.grid(color='gray', linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.045)
    cbar.set_label('Hierarchical mergers in\n Kremer CMC clusters', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)



    ## Sub-solar metallicities ##
    bbh_file = os.path.join(GW24_dir, 'external_cluster_simulations/cBHBd/Z_0.002.dat')
    m1, m2, a1, a2 = np.loadtxt(bbh_file, usecols=(2,3,14,15)).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1

    ### First-generation mergers ###
    m1g = m1[a1+a2==0]
    q1g = q[a1+a2==0]
    a1g = a1[a1+a2==0]
    ### Hierarchical mergers ###
    mng = m1[a1+a2>0]
    qng = q[a1+a2>0]
    ang = a1[a1+a2>0]

    print('Number of 1g+1g mergers in sub-solar metallicity cBHB clusters:', len(m1g))

    # Spins #
    ax = axes[0,0]

    H, xedges, yedges = np.histogram2d(x=np.log10(mng), y=ang, bins=[logm1bins,abins])
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

    ## Add GW241011 and GW241110 posteriors ##
    gw241011_am = np.loadtxt(os.path.join(w_dir,'gw241011_am.txt'))
    s1_bins_241011 = gw241011_am[1:,0]
    log_m1_bins_241011 = gw241011_am[0,1:]
    heights_241011 = gw241011_am[1:,1:]
    c90_241011 = [gw241011_am[0,0]]

    gw241011_am = np.loadtxt(os.path.join(w_dir,'gw241110_am.txt'))
    s1_bins_241110 = gw241110_am[1:,0]
    log_m1_bins_241110 = gw241110_am[0,1:]
    heights_241110 = gw241110_am[1:,1:]
    c90_241110 = [gw241110_am[0,0]]

    ax.contourf(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
    ax.contour(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10)

    ax.contourf(log_m1_bins_241011, s1_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
    ax.contour(log_m1_bins_241011, s1_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11)

    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([])
    ax.set_ylabel(r'$\chi_1$', fontsize=fs)
    ax.set_ylim(ymin=0., ymax=1.0)
    ax.set_yticks(np.linspace(0.,1.0,6))
    #ax.set_box_aspect(1)
    ax.grid(color='gray', linestyle='--')
    ax.set_title(r'$Z\sim 0.1\ \mathrm{Z}_\odot$', fontsize=fs)

    # Masses #
    ax = axes[1,0]

    H, xedges, yedges = np.histogram2d(x=np.log10(m1g), y=q1g, bins=[logm1bins,qbins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    H /= H.sum()
    t = np.linspace(0, H.max(), n)
    integral = ((H >= t[:, None, None]) * H).sum(axis=(1,2))
    contours = interp1d(integral, t)([0.95])
    ax.contour(X, Y, H.T, contours,
               colors=CB_color_cycle[3],
               linestyles='--',
               )
    ax.plot([], [],
            c=CB_color_cycle[3],
            ls='--',
            label='1g+1g',
            )

    print('Number of hierarchical mergers in sub-solar metallicity cBHB clusters:', len(mng))
    H, xedges, yedges = np.histogram2d(x=np.log10(mng), y=qng, bins=[logm1bins,qbins])
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

    ## Add GW241011 and GW241110 posteriors ##
    gw241011_qm = np.loadtxt(os.path.join(w_dir,'gw241011_qm.txt'))
    q_bins_241011 = gw241011_qm[1:,0]
    log_m1_bins_241011 = gw241011_qm[0,1:]
    heights_241011 = gw241011_qm[1:,1:]
    c90_241011 = [gw241011_qm[0,0]]

    gw241110_qm = np.loadtxt(os.path.join(w_dir,'gw241110_qm.txt'))
    q_bins_241110 = gw241110_qm[1:,0]
    log_m1_bins_241110 = gw241110_qm[0,1:]
    heights_241110 = gw241110_qm[1:,1:]
    c90_241110 = [gw241110_qm[0,0]]

    ax.contourf(log_m1_bins_241110, q_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
    ax.contour(log_m1_bins_241110, q_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10)

    ax.contourf(log_m1_bins_241011, q_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
    ax.contour(log_m1_bins_241011, q_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11)

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([10, 30, 100])
    ax.set_ylabel(r'$q$', fontsize=fs)
    ax.set_ylim(ymin=0.1, ymax=1.0)
    ax.set_yticks(np.linspace(0.2,1.0,5))
    #ax.set_box_aspect(1)
    ax.legend(loc='lower left', fontsize=fs)
    ax.grid(color='gray', linestyle='--')


    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'Fig6_cBHB.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'
    grid_dir = os.path.join(FIRE_dir,'grid')
    sim_names = ['m12i','m11d','m11h','m11e','m11q','m11i']

    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical'
    GW24_dir = '/home/tristan-bruel/Documents/Science/o4a-astro/GW241011_GW241110_data_release/data'
    gw241011_color = '#1f78b4'
    gw241110_color = '#33a02c'

    plot_mBH_spin()
    plt.show()
