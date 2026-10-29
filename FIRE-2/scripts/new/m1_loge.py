import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d

import os


def plot_loge_dist():

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
                  '#f781bf', '#a65628', '#e41a1c']

    fig, axes = plt.subplots(1, 2, figsize=(8.,3.5))

    loge_bins = np.linspace(-8, 0, 41)
    m1_bins = np.linspace(0, 100, 101)
    X, Y = np.meshgrid(m1_bins[:-1],loge_bins[:-1])
    cmap = plt.get_cmap('plasma')
    #norm = mpl.colors.Normalize(vmin=1,vmax=100)
    norm = mpl.colors.LogNorm(vmin=1,vmax=2e2)

    mBH1 = np.array([])
    loge = np.array([])
    tej = np.array([])
    w = np.array([])

    ## Use the m12i MHD version of GBoF I & II ##
    print('m12i_MHD')
    sim_dir = os.path.join(FIRE_dir,'m12i_MHD_res7100')
    bbhfile = os.path.join(sim_dir,'bh_dyn.dat')
    teject, m1, m2, ecc, tmerge, weights = np.loadtxt(bbhfile, usecols=[3,4,5,6,7,12], skiprows=1).T
    mBH1 = np.append(mBH1, np.maximum(m1, m2))
    loge = np.append(loge, np.log10(ecc))
    tej = np.append(tej, teject)
    w = np.append(w, weights)

    ## Other FIRE-2 simulations
    bbhfile = os.path.join(FIRE_dir, 'grid/allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    teject, m1, m2, ecc, tmerge = np.loadtxt(bbhfile, usecols=[3,4,5,6,7], skiprows=1).T
    weights = np.ones_like(m1)
    mBH1 = np.append(mBH1, np.maximum(m1, m2))
    loge = np.append(loge, np.log10(ecc))
    tej = np.append(tej, teject)
    w = np.append(w, weights)

    ejected = tej > 0

    ax = axes[0]
    print('Total number of BBH mergers:', len(ecc))
    print('Number of ejected BBH mergers:', np.sum(ejected))

    H_incluster, _, _ = np.histogram2d(
            x=mBH1[~ejected],
            y=loge[~ejected], 
            bins=[m1_bins,loge_bins], 
            weights=w[~ejected],
            )
    print(H_incluster.min(), H_incluster.max())
    ax.pcolormesh(X, Y, H_incluster.T,
                  cmap=cmap,
                  norm=norm,
                  )
    ax.set_title(r'In-cluster')


    ax = axes[1]
    H_ejected, _, _ = np.histogram2d(
            x=mBH1[ejected],
            y=loge[ejected],
            bins=[m1_bins,loge_bins],
            weights=w[ejected],
            )
    ax.pcolormesh(X, Y, H_ejected.T,
                  cmap=cmap,
                  norm=norm,
                  )
    ax.set_title(r'Ejected')

    for ax in axes:
        ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
        ax.set_xticks(np.linspace(0,100,11))
        #ax.set_ylabel(r'$\mathrm{log}_{10}(e/\mathrm{10 Hz})$')
        ax.set_yticks(np.linspace(-8,0,9))
        ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label(r'Number of BBH mergers', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
    cbar.set_ticks([1,10,100])
    cbar.set_ticklabels([r'$1$',r'$10$',r'$10^2$'])


    ## ALL TOGETHER ##
    fig, ax = plt.subplots(1, 1, figsize=(4.,4.))

    H_all, _, _ = np.histogram2d(
            x=mBH1,
            y=loge,
            bins=[m1_bins,loge_bins],
            weights=w,
            )
    ax.pcolormesh(X, Y, H_all.T,
                  cmap=cmap,
                  norm=norm,
                  )
    ax.set_title(r'BBHs from simulated clusters')

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    ax.set_xticks(np.linspace(0,100,11))
    #ax.set_ylabel(r'$\mathrm{log}_{10}(e/\mathrm{10 Hz})$')
    ax.set_yticks(np.linspace(-8,0,9))
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label(r'Number of BBH mergers', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
    cbar.set_ticks([1,10,100])
    cbar.set_ticklabels([r'$1$',r'$10$',r'$10^2$'])



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    plot_loge_dist()
    plt.show()
