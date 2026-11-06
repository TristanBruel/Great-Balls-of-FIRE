import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os
import sys
import argparse




def plot_bbhs_from_hosts(event, FIRE_dir=None, plot_dir=None):

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
    gw_events = np.array(['GW241011', 'GW241110'])
    if event not in gw_events:
        raise ValueError('Event must be part of %s' %gw_events)
    n = np.where(gw_events==event)[0][0]

    ## BBH mergers from all clusters in my simulations ##
    bbhfile = os.path.join(FIRE_dir, 'grid/allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    Z, m1, m2, tmerge, a1, a2 = np.loadtxt(bbhfile, usecols=[1,4,5,7,8,9]).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1, m2), np.minimum(m1,m2)
    qbbh = m2 /m1
    print('Total number of BBH mergers in MSCs: %i' %len(m1))

    fig, axes = plt.subplots(1, 3, figsize=(8.25,2.25), 
            #gridspec_kw={'width_ratios': [1.5, 1]},
            )
    #fig.subplots_adjust(wspace=0.1)

    m1bins = np.linspace(0, 250, 101)
    qbins = np.linspace(0, 1, 26)
    a1bins = np.linspace(0, 1, 26)

    hist, _ = np.histogram(m1, bins=m1bins, 
                           density=True,
                           )
    axes[0].stairs(hist, m1bins,
                   color='gray', alpha=.5, hatch='//',
                   label='All MSCs',
                   )
    hist, _ = np.histogram(qbbh, bins=qbins, 
                           density=True,
                           )
    axes[1].stairs(hist, qbins,
                   color='gray', alpha=.5, hatch='//',
                   )
    hist, _ = np.histogram(a1, bins=a1bins,
                           density=True,
                           )
    axes[2].stairs(hist, a1bins,
                   color='gray', alpha=.5, hatch='//',
                   )

    ## Load cluster ids of identified potential hosts ##
    hosts_file = './log_Bayes_factors/hosts_%s_std.dat' %event
    hosts_id = np.loadtxt(hosts_file, usecols=0, skiprows=1, dtype=str)
    logB = np.loadtxt(hosts_file, usecols=4, skiprows=1)
    tokeep = logB > logB.max()-1
    print('Standard LVK priors')
    print('\tNumber of clusters with logB/logBmax > -1:',np.sum(tokeep))
    print('\tCluster ids with the highest logB:',hosts_id[tokeep])
    print('\tlogB:',logB[tokeep])
    hosts_id=hosts_id[tokeep]
    logB=logB[tokeep]

    loc_bbhs_from_host = []
    Bayes_host = []
    for k,clid in enumerate(hosts_id):
        loc = np.where(cluster_id==clid)[0]
        #hist, _ = np.histogram(m1[loc], bins=m1bins,
        #                       density=True, 
        #                       weights=Bayes_host
        #                       )
        #axes[0].stairs(hist, m1bins,
        #               label=clid,
        #               )
        #hist, _ = np.histogram(qbbh[loc], bins=qbins,
        #                       density=True, 
        #                       weights=Bayes_host
        #                       )
        #axes[1].stairs(hist, qbins,
        #               density=True,
        #               )
        Bayes_host.extend(np.repeat(10**logB[k], len(loc)))
        loc_bbhs_from_host.extend(loc)
    Bayes_host = np.asarray(Bayes_host)
    loc_bbhs_from_host = np.asarray(loc_bbhs_from_host)
    print('\tNumber of BBH mergers found in potential hosts of %s: %i' %(event,len(loc_bbhs_from_host)))

    hist, _ = np.histogram(m1[loc_bbhs_from_host], bins=m1bins, 
                           density=True, 
    #                       weights=Bayes_host
                           )
    axes[0].stairs(hist, m1bins,
                   color=CB_color_cycle[2*n],
                   )
    hist, _ = np.histogram(qbbh[loc_bbhs_from_host], bins=qbins,
                           density=True, 
    #                       weights=Bayes_host
                           )
    axes[1].stairs(hist, qbins,
                   color=CB_color_cycle[2*n],
                   label='Hosts of %s' %event,
                   )
    hist, _ = np.histogram(a1[loc_bbhs_from_host], bins=a1bins,
                           density=True,
    #                       weights=Bayes_host
                           )
    axes[2].stairs(hist, a1bins,
                   color=CB_color_cycle[2*n],
                   )

    """
    ## Load cluster ids of identified potential hosts ##
    hosts_file = './log_Bayes_factors/hosts_%s_astro.dat' %event
    hosts_id = np.loadtxt(hosts_file, usecols=0, skiprows=1, dtype=str)
    logB = np.loadtxt(hosts_file, usecols=4, skiprows=1)
    tokeep = logB > logB.max()-1
    print('Astro priors')
    print('\tNumber of clusters with logB/logBmax > -1:',np.sum(tokeep))
    print('\tCluster ids with the highest logB:',hosts_id[tokeep])
    print('\tlogB:',logB[tokeep])
    hosts_id=hosts_id[tokeep]
    logB=logB[tokeep]

    loc_bbhs_from_host = []
    Bayes_host = []
    for k,clid in enumerate(hosts_id):
        loc = np.where(cluster_id==clid)[0]
        Bayes_host.extend(np.repeat(10**logB[k], len(loc)))
        loc_bbhs_from_host.extend(loc)
    Bayes_host = np.asarray(Bayes_host)
    loc_bbhs_from_host = np.asarray(loc_bbhs_from_host)
    print('\tNumber of BBH mergers found in potential hosts of %s: %i' %(event,len(Bayes_host)))

    hist, _ = np.histogram(m1[loc_bbhs_from_host], bins=m1bins,
                           density=True, #weights=Bayes_host
                           )
    axes[0].stairs(hist, m1bins,
                   color=CB_color_cycle[2*n+1],
                   label='astro prior',
                   )
    hist, _ = np.histogram(qbbh[loc_bbhs_from_host], bins=qbins,
                           density=True, #weights=Bayes_host
                           )
    axes[1].stairs(hist, qbins,
                   color=CB_color_cycle[2*n+1],
                   )
    """

    axes[0].set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    #axes[0].set_xscale('log')
    axes[0].set_xlim(xmin=4, xmax=100)
    axes[0].set_xticks(np.linspace(0,100,6))
    axes[0].set_ylabel(r'$p(m_1)\ [\mathrm{M}_\odot^{-1}]$')
    #axes[0].set_yscale('log')
    #axes[0].set_ylim(ymin=3e-3, ymax=5e1)
    axes[0].legend(loc='upper right', frameon=False)
    axes[0].set_box_aspect(1)

    axes[1].set_xlabel(r'$q$')
    axes[1].set_xlim(xmin=0, xmax=1)
    axes[1].set_xticks(np.linspace(0,1,5))
    #axes[1].tick_params(direction='out', which='both', right=True, left=False, labelleft=False, labelright=True)
    axes[1].set_ylabel(r'$p(q)$')
    #axes[1].yaxis.set_label_position("right") # Move the label to the right
    #axes[1].set_yscale('log')
    #axes[1].set_ylim(ymin=1e0, ymax=2e2)
    #axes[1].set_yticks(np.array([1e1,1e2]))
    axes[1].legend(loc='upper left', frameon=False)
    axes[1].set_box_aspect(1)

    axes[2].set_xlabel(r'$\chi_1$')
    axes[2].set_xlim(xmin=0, xmax=1)
    axes[2].set_xticks(np.linspace(0,1,5))
    #axes[2].tick_params(direction='out', which='both', right=True, left=False, labelleft=False, labelright=True)
    axes[2].set_ylabel(r'$p(\chi_1)$')
    #axes[1].yaxis.set_label_position("right") # Move the label to the right
    #axes[1].set_yscale('log')
    #axes[1].set_ylim(ymin=1e0, ymax=2e2)
    #axes[1].set_yticks(np.array([1e1,1e2]))
    axes[2].set_box_aspect(1)
    
    # Save
    #plt.savefig(os.path.join(plot_dir,'m1_and_q_astro.png'), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir,'bbhs_from_hosts_%s.pdf' %event), format='pdf', bbox_inches='tight')


##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/paper/Figures/Results'

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", help="GW event to consider", nargs='+', default=['GW241011','GW241110'])
    args = parser.parse_args()
    
    for event in args.events:
        print('\nEvent:', event)
        plot_bbhs_from_hosts(event=event, FIRE_dir=FIRE_dir, plot_dir=plot_dir)
    plt.show()
