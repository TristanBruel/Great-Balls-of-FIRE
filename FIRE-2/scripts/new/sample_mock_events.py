import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import argparse

from scipy import stats


def plot_m1_loge_dist(mock_events):

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

    
    fig, ax = plt.subplots(1, 1, figsize=(4.,4.))

    loge_bins = np.linspace(-8, 0, 41)
    m1_bins = np.linspace(0, 100, 101)
    X, Y = np.meshgrid(m1_bins[:-1],loge_bins[:-1])
    cmap = plt.get_cmap('plasma')
    #norm = mpl.colors.Normalize(vmin=1,vmax=100)
    norm = mpl.colors.LogNorm(vmin=1,vmax=1e2)

    H, _, _ = np.histogram2d(
            x=mock_events[:,0],
            y=mock_events[:,2],
            bins=[m1_bins,loge_bins],
            )
    print(H.min(), H.max())
    ax.pcolormesh(X, Y, H.T,
                  cmap=cmap,
                  norm=norm,
                  )
    ax.set_title('Sampled BBHs')

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    ax.set_xticks(np.linspace(0,100,11))
    ax.set_ylabel(r'$\mathrm{log}_{10}(e/\mathrm{10 Hz})$')
    ax.set_yticks(np.linspace(-8,0,9))
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label(r'N BBHs', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
    cbar.set_ticks([1,10,100])
    cbar.set_ticklabels([r'$1$',r'$10$',r'$10^2$'])



def plot_q_loge_dist(dist):

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


    fig, ax = plt.subplots(1, 1, figsize=(4.,4.))

    loge_bins = np.linspace(-8, 0, 41)
    q_bins = np.linspace(0, 1, 101)
    X, Y = np.meshgrid(q_bins[:-1],loge_bins[:-1])
    cmap = plt.get_cmap('plasma')
    #norm = mpl.colors.Normalize(vmin=1,vmax=100)
    norm = mpl.colors.LogNorm(vmin=1,vmax=5e1)

    H, _, _ = np.histogram2d(
            x=mock_events[:,1],
            y=mock_events[:,2],
            bins=[q_bins,loge_bins],
            )
    print(H.min(), H.max())
    ax.pcolormesh(X, Y, H.T,
                  cmap=cmap,
                  norm=norm,
                  )
    ax.set_title('Sampled BBHs')

    ax.set_xlabel(r'$q$')
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel(r'$\mathrm{log}_{10}(e/\mathrm{10 Hz})$')
    ax.set_yticks(np.linspace(-8,0,9))
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label(r'N BBHs', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)
    cbar.set_ticks([1,10,50])
    cbar.set_ticklabels([r'$1$',r'$10$',r'$50$'])


##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    mu = np.array([15,1.0,-5])
    cov = np.array([[200.,0.,0.],[0.,.1,0.],[0.,0.,1.]])
    dist = stats.multivariate_normal(mu, cov)
    mock_events = dist.rvs(10_000)
    m1, q = mock_events[:,0], mock_events[:,1]
    mock_events[:,0][m1<0] = -m1[m1<0]
    mock_events[:,1][q>1] = 1/q[q>1]

    plot_m1_loge_dist(mock_events)
    plot_q_loge_dist(mock_events)
    plt.show()
