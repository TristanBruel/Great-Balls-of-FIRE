import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import argparse

from scipy import stats


def plot_m1_loge_dist(dist):

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

    m1_bins = np.linspace(0, 100, 20)
    q_bins = np.linspace(0, 1, 21)
    loge_bins = np.linspace(-8, 0, 22)
    X, Y, Z = np.meshgrid(m1_bins,q_bins,loge_bins)
    pdf = dist.pdf(np.array([X.flatten(),Y.flatten(),Z.flatten()]).T)
    pdf = pdf.reshape((len(q_bins),len(m1_bins),len(loge_bins)))
    cmap = plt.get_cmap('plasma')
    #norm = mpl.colors.Normalize(vmin=1,vmax=100)
    norm = mpl.colors.Normalize(vmin=0,vmax=pdf.max())

    Y, Z = np.meshgrid(m1_bins,loge_bins)
    data = np.trapezoid(x=q_bins, y=pdf, axis=0)
    ax.contourf(Y, Z, data.T,
                    cmap=cmap,
                    levels=10,
                   )
    ax.contour(Y, Z, data.T, colors='k')
    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    ax.set_xticks(np.linspace(0,100,11))
    ax.set_ylabel(r'$\mathrm{log}_{10}(e/\mathrm{10 Hz})$')
    ax.set_yticks(np.linspace(-8,0,9))
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label(r'pdf', fontsize=fs)
    #cbar.ax.tick_params(labelsize=0.8*fs)
    #cbar.set_ticks([1,10,100])
    #cbar.set_ticklabels([r'$1$',r'$10$',r'$10^2$'])



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

    m1_bins = np.linspace(0, 100, 20)
    q_bins = np.linspace(0, 1, 21)
    loge_bins = np.linspace(-8, 0, 22)
    X, Y, Z = np.meshgrid(m1_bins,q_bins,loge_bins)
    pdf = dist.pdf(np.array([X.flatten(),Y.flatten(),Z.flatten()]).T)
    pdf = pdf.reshape((len(q_bins),len(m1_bins),len(loge_bins)))
    cmap = plt.get_cmap('plasma')
    #norm = mpl.colors.Normalize(vmin=1,vmax=100)
    norm = mpl.colors.Normalize(vmin=0,vmax=pdf.max())

    X, Z = np.meshgrid(q_bins,loge_bins)
    data = np.trapezoid(x=m1_bins, y=pdf, axis=1)
    ax.contourf(X, Z, data.T,
                    cmap=cmap,
                    levels=10,
                   )
    ax.contour(X, Z, data.T, colors='k')
    ax.set_xlabel(r'$q$')
    ax.set_xticks(np.linspace(0,1,11))
    ax.set_ylabel(r'$\mathrm{log}_{10}(e/\mathrm{10 Hz})$')
    ax.set_yticks(np.linspace(-8,0,9))
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_label(r'pdf', fontsize=fs)
    #cbar.ax.tick_params(labelsize=0.8*fs)
    #cbar.set_ticks([1,10,100])
    #cbar.set_ticklabels([r'$1$',r'$10$',r'$10^2$'])


##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'

    mu = np.array([15,1.0,-5])
    cov = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    dist = stats.multivariate_normal(mu, cov)

    plot_m1_loge_dist(dist)
    plot_q_loge_dist(dist)
    plt.show()
