import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d

import os
import argparse


def plot_mBH_spin():

    #####################################
    # Plot your result
    fs = 12
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
    plt.rcParams['legend.fontsize']=.9*fs

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    fig, ax = plt.subplots(1, 1, figsize=(3.5,3.5))

    m1bins = np.linspace(0,200,201)
    qbins = np.linspace(0,1,101)

    bbh_file = os.path.join(grid_dir, 'allBBHs.dat')
    m1, m2, tmerge, a1, a2 = np.loadtxt(bbh_file, usecols=[4,5,7,8,9]).T

    #zmerge = times_to_redshifts(tmerge)
    #astro = zmerge < 1
    #m1 = m1[astro]
    #m2 = m2[astro]
    #a1 = a1[astro]
    #a2 = a2[astro]
    m = np.append(m1,m2)
    a = np.append(a1,a2)
    q = np.minimum(m1,m2) /np.maximum(m1,m2)
    q = np.append(q, q)
    print('Total number of BBH mergers in all FIRE-2 simulations: {}'.format(len(m1)))
    print('Number of hierarchical BBH mergers: {}'.format(np.sum(a1+a2>0)))

    ### First-generation mergers ###
    m1g = m[a==0]
    q1g = q[a==0]
    print('Number of BHs in 1g mergers: {}'.format(len(m1g)))

    H, xedges, yedges = np.histogram2d(x=m1g, y=q1g, bins=[m1bins,qbins], density=True)
    #print(H[H>0].min(), H.max())
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    if Hmin>1:
        vmin = int(Hmin /10**int(np.log10(Hmin)))*10**int(np.log10(Hmin))
    else:
        vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    if Hmax>1:
        vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    else:
        vmax = int(Hmax /10**int(np.log10(Hmax)-1)+1)*10**(int(np.log10(Hmax))-1)
    #print(vmin, vmax)
    H[H==0] = 1e-32

    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    ax_top = ax.inset_axes([0, 1.05, 1, 0.4])
    mhist = np.sum(H *np.diff(qbins), axis=1)
    ax_top.plot(m1bins[:-1], mhist,
                c='forestgreen', label='1g')

    ax_left = ax.inset_axes([-.45, 0., 0.4, 1.], sharey=ax)
    chihist = np.sum(H *np.diff(m1bins)[:,None], axis=0)
    ax_left.plot(chihist, qbins[:-1], c='forestgreen')


    ### Hierarchical mergers ###
    mhierar = m[a>0]
    qhierar = q[a>0]
    print('Number of BHs in hierarchical mergers: {}'.format(len(mhierar)))

    H, xedges, yedges = np.histogram2d(x=mhierar, y=qhierar, bins=[m1bins,qbins], density=True)
    #print(H[H>0].min(), H.max())
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    if Hmin>1:
        vmin = int(Hmin /10**int(np.log10(Hmin)))*10**int(np.log10(Hmin))
    else:
        vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    if Hmax>1:
        vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    else:
        vmax = int(Hmax /10**int(np.log10(Hmax)-1)+1)*10**(int(np.log10(Hmax))-1)
    #print(vmin, vmax)
    H[H==0] = 1e-32

    cmap2 = plt.get_cmap('plasma')
    norm2 = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    im2 = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),5),
                     cmap=cmap2,
                     norm=norm2,
                    )

    mhist = np.sum(H *np.diff(qbins), axis=1)
    ax_top.plot(m1bins[:-1], mhist, c='orangered', label='ng')
    ymin = 1e-3
    ymax = 1e0

    qhist = np.sum(H *np.diff(m1bins)[:,None], axis=0)
    ax_left.plot(qhist, qbins[:-1], c='orangered')

    ax_top.set_xlim(xmin=-5., xmax=100)
    ax_top.set_xticks([])
    ax_top.set_yscale('log')
    ax_top.set_ylim(ymin=ymin, ymax=ymax)
    ax_top.set_yticks([])
    ax_top.set_yticks([], minor=True)
    ax3 = ax_top.twinx()
    ax3.set_ylabel(r'$\mathrm{dn}/\mathrm{d}m_\mathrm{BH}$', fontsize=fs, y=0.5)
    ax3.set_yscale('log')
    ax3.set_ylim(ymin=ymin, ymax=ymax)
    #ax3.set_yticks(np.logspace(-1,3,3))

    ax_left.set_ylabel(r'$q$', fontsize=fs)
    ax_left.set_ylim(ymin=-0.1, ymax=1.0)
    ax_left.set_xlabel(r'$\mathrm{dn}/\mathrm{d}q$', fontsize=fs)
    ax_left.set_xscale('log')
    ax_left.set_xlim(xmin=1e-1, xmax=1e2)

    ax_top.legend(bbox_to_anchor=(0.,-0.1), loc='lower right', fontsize=fs)

    ax.set_xlabel(r'$m_\mathrm{BH}\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(xmin=-5., xmax=100)
    ax.set_xticks(np.linspace(0,100,5))
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_ylim(ymin=-0.1, ymax=1.0)
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045)
    cbar.set_label(r'$\mathrm{d}^2\mathrm{n}/\mathrm{d}m_\mathrm{BH}\ \mathrm{d}q$',
                   fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)


    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'mBH_q_hierarchical_allsims.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'
    grid_dir = os.path.join(FIRE_dir,'grid')
    sim_names = ['m12i','m11d','m11h','m11e','m11q','m11i']

    plot_mBH_spin()
    plt.show()
