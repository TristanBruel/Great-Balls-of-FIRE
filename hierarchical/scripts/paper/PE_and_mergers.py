import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

from plotting_utils import get_contour_heights, Bounded_2d_kde

import os
import argparse


def plot_mBH_spin(events, Kremer=False, cluster=None):

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
    gw_events = np.array(['GW241011', 'GW241110'])
    gw_colors = [gw241011_color, gw241110_color]


    fig, axes = plt.subplots(2, 2, figsize=(6.3,6.))
    fig.subplots_adjust(wspace=0.25)


    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'
    grid_dir = os.path.join(FIRE_dir,'grid')
    if Kremer:
        bbh_file = os.path.join(grid_dir, 'allBBHs_withKremer.dat')
    else:
        bbh_file = os.path.join(grid_dir, 'allBBHs.dat')
    cmc_cluster_ids = np.loadtxt(bbh_file, usecols=0, skiprows=1, dtype=str)
    Z, m1, m2, tmerge, a1, a2 = np.loadtxt(bbh_file, usecols=[1,4,5,7,8,9]).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1
    print('Total number of BBH mergers in CMC clusters: %i' %(len(m1)))
    print('Number of hierarchical BBH mergers: %i' %(np.sum(a1+a2>0)))

    s=1**2
    alpha=0.5
    if cluster:
        in_cluster = cmc_cluster_ids == cluster
        m1 = m1[in_cluster]
        q = q[in_cluster]
        a1 = a1[in_cluster]
        a2 = a2[in_cluster]
        s=3*2
        alpha=1.

    ### First-generation mergers ###
    m1_1g = m1[a1+a2==0]
    q_1g = q[a1+a2==0]
    a1_1g = a1[a1+a2==0]
    ### Hierarchical mergers ###
    m1_ng = m1[a1+a2>0]
    q_ng = q[a1+a2>0]
    a1_ng = a1[a1+a2>0]

    colors = np.zeros(len(m1), dtype='U10')
    colors[a1+a2==0] = 'gray'
    colors[a1+a2>0] = 'firebrick'

    # Primary mass - primary spin #
    for ax in axes[0,:]:
        ax.scatter(np.log10(m1_1g), a1_1g,
                   c='gray', s=s, alpha=alpha,
                   )
        ax.scatter(np.log10(m1_ng), a1_ng,
                   c='firebrick', s=s, alpha=alpha,
                   )
        ax.set_xlim(np.log10(5), np.log10(120))
        ax.set_xticks(np.log10([10, 30, 100]))
        ax.set_xticklabels([])
        ax.set_ylim(ymin=-0.05, ymax=1.0)
        ax.set_yticks(np.linspace(0.,1.0,6))
        ax.set_box_aspect(1)
        ax.grid(color='gray', linestyle='--')

    axes[0,0].set_ylabel(r'$\chi_1$')
    axes[0,1].set_yticklabels([])

    axes[0,0].set_title(r'Standard LVK priors')
    axes[0,1].set_title(r'Astrophysically motivated priors')

    ax.scatter([], [],
               c='gray', s=3**2,
               label='1g+1g mergers'
               )
    ax.scatter([], [],
               color='firebrick', s=3**2,
               label='Hierarchical mergers'
               )
    #axes[0,1].legend(loc='lower right')
    

    # Primary mass - Mass ratio #
    for ax in axes[1,:]:
        ax.scatter(np.log10(m1_1g), q_1g,
                   c='gray', s=s, alpha=alpha,
                   )
        ax.scatter(np.log10(m1_ng), q_ng,
                   c='firebrick', s=s, alpha=alpha,
                   )
        ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
        ax.set_xlim(np.log10(5), np.log10(120))
        ax.set_xticks(np.log10([10, 30, 100]))
        ax.set_xticklabels([10, 30, 100])
        ax.set_ylim(ymin=0., ymax=1.0)
        ax.set_yticks(np.linspace(0.0,1.0,6))
        ax.set_box_aspect(1)
        ax.grid(color='gray', linestyle='--')

    axes[1,0].set_ylabel('$q$')
    axes[1,1].set_yticklabels([])

    # Add GW241011 and GW241110 posteriors #
    legend_elements = []
    for event in events:
        if event not in gw_events:
            raise ValueError('Events must be part of %s' %gw_events)
        n = np.where(gw_events==event)[0][0]
        ## Mass - spin ##
        gw_am = np.loadtxt(os.path.join(w_dir,'scripts/paper/contours',event+'_am.txt'))
        s1_bins = gw_am[1:,0]
        log_m1_bins = gw_am[0,1:]
        heights = gw_am[1:,1:]
        c90 = [gw_am[0,0]]

        ax = axes[0,0]
        ax.contourf(log_m1_bins, s1_bins, heights,
                   levels=[*c90, np.inf], colors=gw_colors[n], zorder=2, alpha=0.5)
        contour_am = ax.contour(log_m1_bins, s1_bins, heights,
                   levels=[*c90, np.inf], colors=gw_colors[n], zorder=2)
        c = contour_am.get_paths()[0]
        mask_am = c.contains_points(np.hstack((np.log10(m1)[:,None],a1[:,None])))

        ### With astro priors ###
        gw_am = np.loadtxt(os.path.join(w_dir,'scripts/paper/contours',event+'_am_astro.txt'))
        s1_bins = gw_am[1:,0]
        log_m1_bins = gw_am[0,1:]
        heights = gw_am[1:,1:]
        c90 = [gw_am[0,0]]

        ax = axes[0,1]
        ax.contourf(log_m1_bins, s1_bins, heights,
                   levels=[*c90, np.inf], colors=gw_colors[n], zorder=2, alpha=0.5)
        contour_am = ax.contour(log_m1_bins, s1_bins, heights,
                   levels=[*c90, np.inf], colors=gw_colors[n], zorder=2)
        c = contour_am.get_paths()[0]
        mask_am_astro = c.contains_points(np.hstack((np.log10(m1)[:,None],a1[:,None])))

        legend_elements.append(
            Patch(color=gw_colors[n], alpha=0.5, label=event),
        )

        ## Mass - mass ratio ##
        gw_qm = np.loadtxt(os.path.join(w_dir,'scripts/paper/contours',event+'_qm.txt'))
        q_bins = gw_qm[1:,0]
        log_m1_bins = gw_qm[0,1:]
        heights = gw_qm[1:,1:]
        c90 = [gw_qm[0,0]]

        ax = axes[1,0]
        ax.contourf(log_m1_bins, q_bins, heights,
                   levels=[*c90, np.inf], colors=gw_colors[n], zorder=2, alpha=0.5)
        contour_qm = ax.contour(log_m1_bins, q_bins, heights,
                   levels=[*c90, np.inf], colors=gw_colors[n], zorder=2)
        c = contour_qm.get_paths()[0]
        mask_qm = c.contains_points(np.hstack((np.log10(m1)[:,None],q[:,None])))

        ### With astro prior ###
        gw_qm = np.loadtxt(os.path.join(w_dir,'scripts/paper/contours',event+'_qm_astro.txt'))
        q_bins = gw_qm[1:,0]
        log_m1_bins = gw_qm[0,1:]
        heights = gw_qm[1:,1:]
        c90 = [gw_qm[0,0]]

        ax = axes[1,1]
        ax.contourf(log_m1_bins, q_bins, heights,
                   levels=[*c90, np.inf], colors=gw_colors[n], zorder=2, alpha=0.5)
        contour_qm = ax.contour(log_m1_bins, q_bins, heights,
                   levels=[*c90, np.inf], colors=gw_colors[n], zorder=2)
        c = contour_qm.get_paths()[0]
        mask_qm_astro = c.contains_points(np.hstack((np.log10(m1)[:,None],q[:,None])))

        ## Show BBH mergers that fall within the PE contours ##
        event_like = mask_am & mask_qm
        print('Number of %s-like mergers: %i' %(event,np.sum(event_like)))
        bbh_file = './contours/bbhs_in_%s.dat' %event
        header='#1:ID #2:Metal #3:M1[Msun] #4:M2[Msun] #5:a1 #6:a2\n'
        with open(bbh_file, 'w') as s:
            s.write(header)
            for i in range(np.sum(event_like)):
                s.write(
                    cmc_cluster_ids[event_like][i] +
                    ' {:.5e} {:.5e} {:.5e} {:.5e} {:.5e}\n'.format(
                        Z[event_like][i],
                        m1[event_like][i],
                        m2[event_like][i],
                        a1[event_like][i],
                        a2[event_like][i])
                )
        axes[0,0].scatter(np.log10(m1[event_like]), a1[event_like],
                          facecolor=gw_colors[n], edgecolor='k', 
                          s=3**2, zorder=3,
                          )
        axes[1,0].scatter(np.log10(m1[event_like]), q[event_like],
                          facecolor=gw_colors[n], edgecolor='k',
                          s=3**2, zorder=3,
                          )

        event_like = mask_am_astro & mask_qm_astro
        print('Number of %s-like mergers with astro priors: %i' %(event,np.sum(event_like)))
        bbh_file = './contours/bbhs_in_%s_astro.dat' %event
        with open(bbh_file, 'w') as s:
            s.write(header)
            for i in range(np.sum(event_like)):
                s.write(
                    cmc_cluster_ids[event_like][i] +
                    ' {:.5e} {:.5e} {:.5e} {:.5e} {:.5e}\n'.format(
                        Z[event_like][i],
                        m1[event_like][i],
                        m2[event_like][i],
                        a1[event_like][i],
                        a2[event_like][i])
                )
        axes[0,1].scatter(np.log10(m1[event_like]), a1[event_like],
                          facecolor=gw_colors[n], edgecolor='k',
                          s=3**2, zorder=3,
                          label='Events in\nPE contours',
                          )
        axes[1,1].scatter(np.log10(m1[event_like]), q[event_like],
                          facecolor=gw_colors[n], edgecolor='k',
                          s=3**2, zorder=3,
                          )


    axes[0,1].legend(loc='lower right')
    axes[0,0].legend(handles=legend_elements, loc='lower right')

    # Save
    plot_dir = os.path.join(w_dir,'plots/paper')
    name = '_'.join(events)
    if cluster:
        filename = 'PE_%s_and_mergers' %name +Kremer*'_withKremer'+'_%s'%cluster+'.pdf'
    else:
        filename = 'PE_%s_and_mergers' %name +Kremer*'_withKremer'+'.pdf'
    plt.savefig(os.path.join(plot_dir,filename), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":
    

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", help="GW event to consider", nargs='+', default=['GW241011', 'GW241110']) 
    parser.add_argument("--Kremer", dest='Kremer', action='store_true', help="Include Kremer CMC clusters")
    parser.add_argument("--cluster", dest='cluster', help="Id of the cluster for which to show BBH mergers", type=str, default=None)
    args = parser.parse_args()

    plot_mBH_spin(events=args.events, Kremer=args.Kremer, cluster=args.cluster)
    plt.show()
