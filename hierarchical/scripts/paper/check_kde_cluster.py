import numpy as np
import h5py
from kaydee import KDE

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

import os
import argparse


def plot_KDE(cluster=None):

    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical'

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
    plt.rcParams['legend.fontsize']=.9*fs
    plt.rcParams['axes.titlesize']=fs

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    gw241011_color = '#1f78b4'
    gw241110_color = '#33a02c'
    gw_colors = [gw241011_color, gw241110_color]


    fig, axes = plt.subplots(1, 2, figsize=(10.,4.))
    fig.subplots_adjust(wspace=0.1)

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'
    grid_dir = os.path.join(FIRE_dir,'grid')
    bbh_file = os.path.join(grid_dir, 'allBBHs.dat')
    cmc_cluster_ids = np.loadtxt(bbh_file, usecols=0, skiprows=1, dtype=str)
    Z, m1, m2, tmerge, a1, a2 = np.loadtxt(bbh_file, usecols=[1,4,5,7,8,9]).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1
    print('Total number of BBH mergers in CMC clusters: %i' %(len(m1)))
    print('Number of hierarchical BBH mergers: %i' %(np.sum(a1+a2>0)))
    m1_grid = np.linspace(0,210,101)
    q_grid = np.linspace(0,1,51)
    a1_grid = np.linspace(0,1,21)
    X, Y, Z = np.meshgrid(m1_grid, q_grid, a1_grid)
    grid = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    in_cluster = cmc_cluster_ids == cluster
    m1 = m1[in_cluster]
    q = q[in_cluster]
    a1 = a1[in_cluster]
    a2 = a2[in_cluster]

    kde = KDE([m1, q, a1])
    pdf = kde.pdf(grid.T).reshape(len(q_grid),len(m1_grid),len(a1_grid))

    # Primary mass - primary spin #
    ax=axes[0]
    
    cmap=plt.get_cmap('plasma')
    norm = mpl.colors.Normalize(vmin=-10, vmax=-1)
    pm1pq = np.trapezoid(x=a1_grid, y=pdf, axis=2)
    c = ax.pcolormesh(m1_grid, q_grid, np.log(pm1pq),
                      cmap=cmap, norm=norm,
                     )

    ax.scatter(m1, q,
               marker='x', c='k', s=3**2,
               )

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    ax.set_xlim(0,100)
    ax.set_ylabel(r'$q$')
    ax.set_ylim(0,1)
    ax.set_box_aspect(1)


    # Primary mass - primary spin #
    ax=axes[1]
    pm1pa1 = np.trapezoid(x=q_grid, y=pdf, axis=0)
    c = ax.pcolormesh(m1_grid, a1_grid, np.log(pm1pa1).T,
                      cmap=cmap, norm=norm,
                     )

    ax.scatter(m1, a1,
               marker='x', c='k', s=3**2,
               )

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    ax.set_xlim(0,100)
    ax.set_ylabel(r'$\chi_1$')
    ax.set_ylim(0,1)
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.045)
    cbar.set_label(r'log-KDE')
    cbar.ax.tick_params(labelsize=0.8*fs)

    title = 'cluster ' + ' '.join(cluster.split('_'))
    fig.suptitle(title)

    # Save
    plot_dir = os.path.join(w_dir,'plots/paper')
    filename = 'KDE_%s'%cluster+'.pdf'
    plt.savefig(os.path.join(plot_dir,filename), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":
    

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", dest='cluster', help="Id of the cluster for which to show BBH mergers", type=str, default=None)
    args = parser.parse_args()

    plot_KDE(cluster=args.cluster)
    plt.show()
