import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

from plotting_utils import get_contour_heights, Bounded_2d_kde

import os
import argparse



def plot_clusters_Mcl_rhm(cluster, Kremer=False, plot_dir=None):

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
    color_all = CB_color_cycle[6] # gray
    color_hierar = CB_color_cycle[8] # yellow
    color_inPE = CB_color_cycle[0] # blue
    color_inPE_astro = CB_color_cycle[7] # red

    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'
    cluster_file = os.path.join(grid_dir, 'gridClusters'+Kremer*'_withKremer'+'.hdf5')
    Zsun = 0.017
    with h5py.File(cluster_file) as c:
        cluster_id = c['id'][()]
        cluster_mass = c['Mass'][()]
        cluster_radius = c['HalfMassRadius'][()]
        cluster_Z = c['Metallicity'][()] /Zsun
        cluster_nBBHs = c['nBBHs'][()]
    cluster_id = np.astype(cluster_id, str)
    logMcl = np.log10(cluster_mass)
    logRhm = np.log10(cluster_radius)

    ### Identify clusters that host hierarchical BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs'+Kremer*'_withKremer'+'.dat')
    cmc_cluster_ids = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    m1, m2 = np.maximum(m1, m2), np.minimum(a1,a2)
    hierar = a1+a2>0
    unique, indexes = np.unique(cmc_cluster_ids[hierar], return_index=True)
    host_hierar = np.array([np.where(cluster_id==u)[0][0] for u in unique])

    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/scripts/paper/contours'
    logm_bins = np.linspace(4, 7, 31)
    midm_bins = (logm_bins[1:]+logm_bins[:-1])/2
    logr_bins = np.linspace(-3, 3, 61)
    midr_bins = (logr_bins[1:]+logr_bins[:-1])/2

    fig, ax = plt.subplots(1, 1, figsize=(2.7,2.7))

    H, xedges, yedges = np.histogram2d(x=logMcl,
                                       y=logRhm,
                                       bins=[logm_bins,logr_bins],
                                       density=True,
                                       )
    logvmin = (int(10*np.log10(H[H>0].min()))-1) /10
    logvmax = (int(10*np.log10(H[H>0].max()))+1) /10
    H[H<1e-32] = 1e-32

    ax.scatter(logMcl, logRhm,
               color=color_all, marker='o', s=3**2,
               zorder=0,
               label='all MSCs'
               )

    ## Clusters that host hierarchical mergers ##
    H_hierar, xedges, yedges = np.histogram2d(x=logMcl[host_hierar],
                                              y=logRhm[host_hierar],
                                              bins=[logm_bins,logr_bins],
                                              density=True,
                                              )
    ax.scatter(logMcl[host_hierar], logRhm[host_hierar],
               marker='*', s=6**2, linewidths=1.5,
               edgecolor=color_hierar,
               zorder=1,
               label='Hosts of hierarchical mergers'
               )

    if cluster:
        loc = np.where(cluster_id==cluster)[0][0]
        ax.scatter(logMcl[loc], logRhm[loc],
               marker='o', s=6**2, linewidths=1.5,
               zorder=2, color='firebrick',
               label=cluster
               )


    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$')
    ax.set_xlim(xmin=4.5, xmax=7)
    ax.set_xticks(np.linspace(5,7,3))
    ax.set_ylim(ymin=-3., ymax=3.)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.legend(loc='lower right')
    ax.set_box_aspect(1)



def plot_clusters_Mcl_Z(cluster=None, Kremer=False, plot_dir=None):

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
    color_all = CB_color_cycle[6] # gray
    color_hierar = CB_color_cycle[8] # yellow
    color_inPE = CB_color_cycle[0] # blue
    color_inPE_astro = CB_color_cycle[7] # red


    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'
    cluster_file = os.path.join(grid_dir, 'gridClusters'+Kremer*'_withKremer'+'.hdf5')
    Zsun = 0.017
    with h5py.File(cluster_file) as c:
        cluster_id = c['id'][()]
        cluster_mass = c['Mass'][()]
        cluster_radius = c['HalfMassRadius'][()]
        cluster_Z = c['Metallicity'][()] /Zsun
        cluster_nBBHs = c['nBBHs'][()]
    cluster_id = np.astype(cluster_id, str)
    logMcl = np.log10(cluster_mass)
    logRhm = np.log10(cluster_radius)
    logZ = np.log10(cluster_Z)

    ### Identify clusters that host hierarchical BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs'+Kremer*'_withKremer'+'.dat')
    cmc_cluster_ids = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    m1, m2 = np.maximum(m1, m2), np.minimum(a1,a2)
    hierar = a1+a2>0
    unique, indexes = np.unique(cmc_cluster_ids[hierar], return_index=True)
    host_hierar = np.array([np.where(cluster_id==u)[0][0] for u in unique])


    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/scripts/paper/contours'
    logm_bins = np.linspace(4, 7, 31)
    midm_bins = (logm_bins[1:]+logm_bins[:-1])/2
    logZ_bins = np.linspace(-3, 1, 41)
    midZ_bins = (logZ_bins[1:]+logZ_bins[:-1])/2

    fig, ax = plt.subplots(1, 1, figsize=(2.7,2.7))

    H, xedges, yedges = np.histogram2d(
            x=logMcl, 
            y=logZ, 
            bins=[logm_bins,logZ_bins],
            density=True,
    )
    logvmin = (int(10*np.log10(H[H>0].min()))-1) /10
    logvmax = (int(10*np.log10(H[H>0].max()))+1) /10
    H[H<1e-32] = 1e-32

    ax.scatter(logMcl, logZ,
               color=color_all, marker='o', s=3**2,
               zorder=0,
               #label='all MSCs'
               )

    ## Clusters that host hierarchical mergers ##
    H_hierar, xedges, yedges = np.histogram2d(
            x=logMcl[host_hierar], 
            y=logZ[host_hierar], 
            bins=[logm_bins,logZ_bins],
            density=True,
    )
    ax.scatter(logMcl[host_hierar], logZ[host_hierar],
               marker='*', s=6**2, linewidths=1.5,
               edgecolor=color_hierar,
               zorder=1,
               #label='Hosts of hierar. mergers'
               )

    if cluster:
        loc = np.where(cluster_id==cluster)[0][0]
        ax.scatter(logMcl[loc], logZ[loc],
               marker='o', s=6**2, linewidths=1.5,
               zorder=2, color='firebrick',
               label=cluster
               )


    ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$')
    ax.set_xlim(xmin=4.5, xmax=7)
    ax.set_xticks(np.linspace(5,7,3))
    ax.set_ylim(ymin=-3., ymax=1.)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_box_aspect(1)



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots/paper'

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", help="Cluster to show", type=str)
    parser.add_argument("--Kremer", action='store_true', help="Include CMC clusters from Kremer et al. 2018")
    args = parser.parse_args()

    plot_clusters_Mcl_Z(cluster=args.cluster, 
                        Kremer=args.Kremer,
                        plot_dir=plot_dir,
                        )
    plot_clusters_Mcl_rhm(cluster=args.cluster, 
                        Kremer=args.Kremer,
                        plot_dir=plot_dir,
                        )
    plt.show()
