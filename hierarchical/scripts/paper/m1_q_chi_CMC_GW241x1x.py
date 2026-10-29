import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

from get_PE_contourf import eval_kde_m1_q, eval_kde_m1_s1

import os
import argparse


def plot_mBH_spin(Zsun=0.017, plot_dir=None, Kremer=False, astro_prior=False):

    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rcParams['font.size']=fs
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='cmr10'
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.use_mathtext']=True
    plt.rcParams['lines.linewidth']=lw
    plt.rcParams['xtick.labelsize']=fs
    plt.rcParams['ytick.labelsize']=fs
    plt.rcParams['legend.fontsize']=.85*fs
    plt.rcParams['axes.titlesize']=fs

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    fig, axes = plt.subplots(2, 3, figsize=(8.,3.5))
    fig.subplots_adjust(wspace=0.1)

    abins = np.linspace(0.,1.0,51)
    logm1bins = np.linspace(0.1,2.1,41)
    qbins = np.linspace(0.1,1.,19)

    if Kremer:
        bbh_file = os.path.join(grid_dir, 'allBBHs_withKremer.dat')
    else:
        bbh_file = os.path.join(grid_dir, 'allBBHs.dat')
    Z, m1, m2, tmerge, a1, a2 = np.loadtxt(bbh_file, usecols=[1,4,5,7,8,9]).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1
    print('Total number of BBH mergers in CMC clusters: %i' %(len(m1)))
    print('Number of hierarchical BBH mergers: %i' %(np.sum(a1+a2>0)))
    primary_1g = ((a1>0)&(m1<m2)) | ((a2>0)&(m2<m1))
    print('Number of hierarchical mergers where primary is 1g: %i' %np.sum(primary_1g))
    GW24x = (m1 < 30) & (q < 0.85) & (a1 > 0.05)

    ## Set colorbar limits and end
    H, xedges, yedges = np.histogram2d(x=np.log10(m1), y=a1, bins=[logm1bins,abins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    ### First-generation mergers ###
    m1g = m1[a1+a2==0]
    q1g = q[a1+a2==0]
    Z1g = Z[a1+a2==0]
    ### Hierarchical mergers ###
    mng = m1[a1+a2>0]
    qng = q[a1+a2>0]
    Zng = Z[a1+a2>0]
    ang = a1[a1+a2>0]


    ## Zero-ish metallicities ##
    # Spins #
    ax = axes[0,0]

    #tokeep = np.abs(np.log10(Zng/(0.01*Zsun))) < 0.25
    tokeep = np.log10(Zng/Zsun) < -1
    print('Number of hierarchical mergers in zero-ish metallicity clusters:', np.sum(tokeep))
    if np.sum(tokeep)>0:
        H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=ang[tokeep], bins=[logm1bins,abins])
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
    gw241011_am = np.loadtxt(os.path.join(w_dir,'gw241011_am'+astro_prior*'_astro'+'.txt'))
    s1_bins_241011 = gw241011_am[1:,0]
    log_m1_bins_241011 = gw241011_am[0,1:]
    heights_241011 = gw241011_am[1:,1:]
    c90_241011 = [gw241011_am[0,0]]

    gw241110_am = np.loadtxt(os.path.join(w_dir,'gw241110_am'+astro_prior*'_astro'+'.txt'))
    s1_bins_241110 = gw241110_am[1:,0]
    log_m1_bins_241110 = gw241110_am[0,1:]
    heights_241110 = gw241110_am[1:,1:]
    c90_241110 = [gw241110_am[0,0]]

    ax.contourf(log_m1_bins_241011, s1_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=10, alpha=0.5)
    ax.contour(log_m1_bins_241011, s1_bins_241110, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=10)
    patch1 = Patch(color=gw241011_color, alpha=0.5)

    ax.contourf(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=11, alpha=0.5)
    ax.contour(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=11)
    patch2 = Patch(color=gw241110_color, alpha=0.5)

    #ax.fill_between(log_m1_bins_241011, y1=0.05, y2=1., where=log_m1_bins_241011<np.log10(30),
    #                facecolor="none", hatch="/", edgecolor="gray",
    #                )

    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([])
    ax.set_ylabel('$\chi_1$')
    ax.set_ylim(ymin=0., ymax=1.0)
    ax.set_yticks(np.linspace(0.,1.0,6))
    #ax.set_box_aspect(1)
    ax.legend(handles=(patch1,patch2), labels=('GW241011','GW241110'),
              loc='lower right',
              )
    ax.grid(color='gray', linestyle='--')
    ax.set_title('$Z\\sim 0.01\\ \\rm{Z}_\\odot$')


    # Masses #
    ax = axes[1,0]

    #tokeep_1g = np.abs(np.log10(Z1g/(0.01*Zsun))) < 0.25
    tokeep_1g = np.log10(Z1g/Zsun) < -1
    print('Number of 1g+1g mergers in zero-ish metallicity clusters:', np.sum(tokeep_1g))
    if np.sum(tokeep_1g)>0:
        #H, xedges, yedges = np.histogram2d(x=np.log10(m1g[tokeep_1g]), y=q1g[tokeep_1g], bins=[logm1bins,qbins])
        #X, Y = np.meshgrid(xedges[1:], yedges[1:])
        #H = (H-H.min()) /(H.max()-H.min())
        #ax.contour(X, Y, H.T,
        #           levels=[0.05], origin='lower',
        #           colors=CB_color_cycle[3], linestyles='--',
        #           )
        log_m1_bins, q_bins, heights, c90 \
                = eval_kde_m1_q(np.log10(m1g[tokeep_1g]), q1g[tokeep_1g], nbins=(50, 49), probs=(0.90,))
        ax.contour(log_m1_bins, q_bins, heights.reshape(q_bins.size, log_m1_bins.size),
                   levels=[*c90], 
                   colors=CB_color_cycle[3], linestyles='--',
                   )
    ax.plot([], [],
            c=CB_color_cycle[3],
            ls='--',
            label='1g+1g',
            )

    if np.sum(tokeep)>0:
        H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=qng[tokeep], bins=[logm1bins,qbins])
        X, Y = np.meshgrid(xedges[1:], yedges[1:])
        Hmin=H[H>0].min()
        Hmax=H[H>0].max()
        vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
        vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
        H[H==0] = 1e-32
        #im = ax.contourf(X, Y, H.T,
        #                 levels=np.logspace(np.log10(vmin),np.log10(vmax),10),
        #                 cmap=cmap,
        #                 norm=norm,
        #                )

    
    ## Add GW241011 and GW241110 posteriors ##
    gw241011_qm = np.loadtxt(os.path.join(w_dir,'gw241011_qm'+astro_prior*'_astro'+'.txt'))
    q_bins_241011 = gw241011_qm[1:,0]
    log_m1_bins_241011 = gw241011_qm[0,1:]
    heights_241011 = gw241011_qm[1:,1:]
    c90_241011 = [gw241011_qm[0,0]]

    gw241110_qm = np.loadtxt(os.path.join(w_dir,'gw241110_qm'+astro_prior*'_astro'+'.txt'))
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

    #ax.fill_between(log_m1_bins_241011, y1=0., y2=0.85, where=log_m1_bins_241011<np.log10(30),
    #                facecolor="none", hatch="/", edgecolor="gray",
    #                )

    ax.set_xlabel('$m_1\\ [\\rm{M}_\\odot]$')
    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([10, 30, 100])
    ax.set_ylabel('$q$')
    ax.set_ylim(ymin=0.1, ymax=1.0)
    ax.set_yticks(np.linspace(0.2,1.0,5))
    #ax.set_box_aspect(1)
    ax.legend(loc='lower right')
    ax.grid(color='gray', linestyle='--')

    
    ## Sub-solar metallicities ##
    # Spins #
    ax = axes[0,1]

    tokeep = np.abs(np.log10(Zng/(0.1*Zsun))) < 0.25
    print('Number of hierarchical mergers in sub-solar metallicity clusters:', np.sum(tokeep))
    if np.sum(tokeep)>0:
        H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=ang[tokeep], bins=[logm1bins,abins])
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
    s1_bins_241011 = gw241011_am[1:,0]
    log_m1_bins_241011 = gw241011_am[0,1:]
    heights_241011 = gw241011_am[1:,1:]
    c90_241011 = [gw241011_am[0,0]]

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

    #ax.fill_between(log_m1_bins_241011, y1=0.05, y2=1., where=log_m1_bins_241011<np.log10(30),
    #                facecolor="none", hatch="/", edgecolor="gray",
    #                )

    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([])
    #ax.set_ylabel('$\chi_1$')
    ax.set_ylim(ymin=0., ymax=1.0)
    ax.set_yticks(np.linspace(0.,1.0,6))
    ax.set_yticklabels([])
    #ax.set_box_aspect(1)
    ax.grid(color='gray', linestyle='--')
    ax.set_title('$Z\\sim 0.1\\ \\rm{Z}_\\odot$')


    # Masses #
    ax = axes[1,1]

    tokeep_1g = np.abs(np.log10(Z1g/(0.1*Zsun))) < 0.25
    print('Number of 1g+1g mergers in sub-solar metallicity clusters:', np.sum(tokeep_1g))
    if np.sum(tokeep_1g):
        #H, xedges, yedges = np.histogram2d(x=np.log10(m1g[tokeep_1g]), y=q1g[tokeep_1g], bins=[logm1bins,qbins])
        #X, Y = np.meshgrid(xedges[1:], yedges[1:])
        #H = (H-H.min()) /(H.max()-H.min())
        #ax.contour(X, Y, H.T,
        #           levels=[0.05], origin='lower',
        #           colors=CB_color_cycle[3], linestyles='--',
        #           )
        log_m1_bins, q_bins, heights, c90 \
                = eval_kde_m1_q(np.log10(m1g[tokeep_1g]), q1g[tokeep_1g], nbins=(50, 49), probs=(0.90,))
        ax.contour(log_m1_bins, q_bins, heights.reshape(q_bins.size, log_m1_bins.size),
                   levels=[*c90],
                   colors=CB_color_cycle[3], linestyles='--',
                   )
    ax.plot([], [],
            c=CB_color_cycle[3],
            ls='--',
            )

    if np.sum(tokeep)>0:
        H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=qng[tokeep], bins=[logm1bins,qbins])
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
    q_bins_241011 = gw241011_qm[1:,0]
    log_m1_bins_241011 = gw241011_qm[0,1:]
    heights_241011 = gw241011_qm[1:,1:]
    c90_241011 = [gw241011_qm[0,0]]

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

    #ax.fill_between(log_m1_bins_241011, y1=0., y2=0.85, where=log_m1_bins_241011<np.log10(30),
    #                facecolor="none", hatch="/", edgecolor="gray",
    #                )

    ax.set_xlabel('$m_1\\ [\\rm{M}_\\odot]$')
    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([10, 30, 100])
    #ax.set_ylabel('$q$')
    ax.set_ylim(ymin=0.1, ymax=1.0)
    ax.set_yticks(np.linspace(0.2,1.0,5))
    ax.set_yticklabels([])
    #ax.set_box_aspect(1)
    ax.grid(color='gray', linestyle='--')


    ## Solar metallicities ##
    # Spins #
    ax = axes[0,2]

    tokeep = np.abs(np.log10(Zng/Zsun)) < 0.25
    print('Number of hierarchical mergers in solar metallicity clusters:', np.sum(tokeep))
    if np.sum(tokeep)>0:
        H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=ang[tokeep], bins=[logm1bins,abins])
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
    s1_bins_241011 = gw241011_am[1:,0]
    log_m1_bins_241011 = gw241011_am[0,1:]
    heights_241011 = gw241011_am[1:,1:]
    c90_241011 = [gw241011_am[0,0]]

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

    
    #ax.fill_between(log_m1_bins_241011, y1=0.05, y2=1., where=log_m1_bins_241011<np.log10(30),
    #                facecolor="none", hatch="/", edgecolor="gray",
    #                )


    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([])
    #ax.set_ylabel('$\chi_1$')
    ax.set_ylim(ymin=0., ymax=1.0)
    ax.set_yticks(np.linspace(0.,1.0,6))
    ax.set_yticklabels([])
    #ax.set_box_aspect(1)
    ax.grid(color='gray', linestyle='--')
    ax.set_title(r'$Z\sim\mathrm{Z}_\odot$')

    # Masses #
    ax = axes[1,2]

    tokeep_1g = np.abs(np.log10(Z1g/Zsun)) < 0.25
    print('Number of 1g+1g mergers in solar metallicity clusters:', np.sum(tokeep_1g))
    if np.sum(tokeep_1g)>0:
        #H, xedges, yedges = np.histogram2d(x=np.log10(m1g[tokeep_1g]), y=q1g[tokeep_1g], bins=[logm1bins,qbins])
        #X, Y = np.meshgrid(xedges[1:], yedges[1:])
        #H = (H-H.min()) /(H.max()-H.min())
        #ax.contour(X, Y, H.T, 
        #           levels=[0.05], origin='lower',
        #           colors=CB_color_cycle[3], linestyles='--',
        #           )
        log_m1_bins, q_bins, heights, c90 \
                = eval_kde_m1_q(np.log10(m1g[tokeep_1g]), q1g[tokeep_1g], nbins=(50, 49), probs=(0.90,))
        ax.contour(log_m1_bins, q_bins, heights.reshape(q_bins.size, log_m1_bins.size),
                   levels=[*c90],
                   colors=CB_color_cycle[3], linestyles='--',
                   )

    
    if np.sum(tokeep)>0:
        H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=qng[tokeep], bins=[logm1bins,qbins])
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
    q_bins_241011 = gw241011_qm[1:,0]
    log_m1_bins_241011 = gw241011_qm[0,1:]
    heights_241011 = gw241011_qm[1:,1:]
    c90_241011 = [gw241011_qm[0,0]]

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

    #ax.fill_between(log_m1_bins_241011, y1=0., y2=0.85, where=log_m1_bins_241011<np.log10(30),
    #                facecolor="none", hatch="/", edgecolor="gray",
    #                )

    ax.set_xlabel('$m_1\\ [\\rm{M}_\\odot]$')
    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([10, 30, 100])
    #ax.set_ylabel('$q$')
    ax.set_ylim(ymin=0.1, ymax=1.0)
    ax.set_yticks(np.linspace(0.2,1.0,5))
    ax.set_yticklabels([])
    #ax.set_box_aspect(1)
    ax.grid(color='gray', linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.045, pad=0.02)
    cbar.set_label('Hierarchical mergers in CMC clusters')
    cbar.ax.tick_params(labelsize=0.8*fs)


    # Save
    filename = 'm1_q_chi_CMC'+Kremer*'_withKremer'+'_gw241x1x'+astro_prior*'_astro'+'.pdf'
    plt.savefig(os.path.join(plot_dir,filename), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":
    
    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical'
    plot_dir = os.path.join(w_dir,'plots/paper')

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'
    grid_dir = os.path.join(FIRE_dir,'grid')
    sim_names = ['m12i','m11d','m11h','m11e','m11q','m11i']

    gw241011_color = '#1f78b4'
    gw241110_color = '#33a02c'

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--astro", dest='astro_prior', action='store_true', help="Uses PE samples with astro priors")
    parser.add_argument("--Kremer", dest='Kremer', action='store_true', help="Include Kremer CMC clusters")
    args = parser.parse_args()

    plot_mBH_spin(plot_dir=plot_dir, Kremer=args.Kremer, astro_prior=args.astro_prior)
    plt.show()
