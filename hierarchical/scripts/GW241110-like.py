import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

import os


def plot_mBH_spin_q(Zsun=0.017, n=1000):

    # GW241011 posteriors #
    gw241011_qm = np.loadtxt(os.path.join(w_dir,'gw241011_qm.txt'))
    q_bins_241011 = gw241011_qm[1:,0]
    log_m1_bins_241011 = gw241011_qm[0,1:]
    heights_241011_q = gw241011_qm[1:,1:]
    c90_241011_q = [gw241011_qm[0,0]]
    gw241011_am = np.loadtxt(os.path.join(w_dir,'gw241011_am.txt'))
    s1_bins_241011 = gw241011_am[1:,0]
    heights_241011_a = gw241011_am[1:,1:]
    c90_241011_a = [gw241011_am[0,0]]

    # GW241110 posteriors #
    gw241110_qm = np.loadtxt(os.path.join(w_dir,'gw241110_qm.txt'))
    q_bins_241110 = gw241110_qm[1:,0]
    log_m1_bins_241110 = gw241110_qm[0,1:]
    heights_241110_q = gw241110_qm[1:,1:]
    c90_241110_q = [gw241110_qm[0,0]]
    gw241110_am = np.loadtxt(os.path.join(w_dir,'gw241110_am.txt'))
    s1_bins_241110 = gw241110_am[1:,0]
    heights_241110_a = gw241110_am[1:,1:]
    c90_241110_a = [gw241110_am[0,0]]

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

    abins = np.linspace(0.,1.0,51)
    logm1bins = np.linspace(0.1,2.1,41)
    qbins = np.linspace(0.1,1.,19)

    bbh_file = os.path.join(grid_dir, 'allBBHs.dat')
    Z, m1, m2, tmerge, a1, a2 = np.loadtxt(bbh_file, usecols=[1,4,5,7,8,9]).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1
    print('Total number of BBH mergers in my CMC clusters: {}'.format(len(m1)))
    print('Number of hierarchical BBH mergers: {}\n'.format(np.sum(a1+a2>0)))

    # Spins #
    for ax in axes[0,:]:
    
        ax.contourf(log_m1_bins_241110, s1_bins_241110, heights_241110_a,
                   levels=[*c90_241110_a, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
        contour_a = ax.contour(log_m1_bins_241110, s1_bins_241110, heights_241110_a,
                   levels=[*c90_241110_a, np.inf], colors=gw241110_color, zorder=10)
        patch1 = Patch(color=gw241110_color, alpha=0.5)

        ax.contourf(log_m1_bins_241011, s1_bins_241011, heights_241011_a,
                   levels=[*c90_241011_a, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
        ax.contour(log_m1_bins_241011, s1_bins_241011, heights_241011_a,
                   levels=[*c90_241011_a, np.inf], colors=gw241011_color, zorder=11)
        patch2 = Patch(color=gw241011_color, alpha=0.5)


        ax.set_xlim(np.log10(5), np.log10(120))
        ax.set_xticks(np.log10([10, 30, 100]))
        ax.set_xticklabels([])
        #ax.set_ylabel(r'$\chi_1$', fontsize=fs)
        ax.set_ylim(ymin=0., ymax=1.0)
        ax.set_yticks(np.linspace(0.,1.0,6))
        #ax.set_box_aspect(1)
        ax.grid(color='gray', linestyle='--')

    axes[0,0].set_ylabel(r'$\chi_1$', fontsize=fs)
    axes[0,0].set_yticks(np.linspace(0.,1.0,6))
    axes[0,1].set_yticklabels([])

    # Masses #
    for ax in axes[1,:]:

        ax.contourf(log_m1_bins_241110, q_bins_241110, heights_241110_q,
                   levels=[*c90_241110_q, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
        contour_q = ax.contour(log_m1_bins_241110, q_bins_241110, heights_241110_q,
                   levels=[*c90_241110_q, np.inf], colors=gw241110_color, zorder=10)

        ax.contourf(log_m1_bins_241011, q_bins_241011, heights_241011_q,
                   levels=[*c90_241011_q, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
        ax.contour(log_m1_bins_241011, q_bins_241011, heights_241011_q,
                   levels=[*c90_241011_q, np.inf], colors=gw241011_color, zorder=11)


        ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$', fontsize=fs)
        ax.set_xlim(np.log10(5), np.log10(120))
        ax.set_xticks(np.log10([10, 30, 100]))
        ax.set_xticklabels([10, 30, 100])
        #ax.set_ylabel(r'$q$', fontsize=fs)
        ax.set_ylim(ymin=0.1, ymax=1.0)
        ax.set_yticks(np.linspace(0.2,1.0,5))
        #ax.set_box_aspect(1)
        ax.grid(color='gray', linestyle='--')

    axes[1,0].set_ylabel(r'$q$', fontsize=fs)
    axes[1,0].set_yticks(np.linspace(0.2,1.0,5))
    axes[1,1].set_yticklabels([])


    ## Solar metallicities ##
    tokeep = np.abs(np.log10(Z/Zsun)) < 0.25
    print('Number of BBH mergers in solar-like metallicity clusters: %i' %np.sum(tokeep))
    p = contour_q.get_paths()[0]
    mask_q = p.contains_points(np.hstack((np.log10(m1)[tokeep,np.newaxis],q[tokeep,np.newaxis])))
    p = contour_a.get_paths()[0]
    mask_a = p.contains_points(np.hstack((np.log10(m1)[tokeep,np.newaxis],a1[tokeep,np.newaxis])))
    GW241110 = mask_q & mask_a
    print('Number of GW241110-like mergers: %i \n' %np.sum(GW241110))
    axes[0,1].scatter(np.log10(m1[tokeep][GW241110]), a1[tokeep][GW241110], 
                      c='darkorange', s=5**2, label='my sims',
                      )
    axes[1,1].scatter(np.log10(m1[tokeep][GW241110]), q[tokeep][GW241110], 
                      c='darkorange', s=5**2, label='my sims',
                      )
    axes[0,1].set_title(r'$Z\sim\ \mathrm{Z}_\odot$', fontsize=fs)
    h, l = axes[0,1].get_legend_handles_labels()
    handles = [patch1,patch2] + h
    labels = ['GW241110','GW241011'] + l
    axes[0,1].legend(handles=handles, labels=labels,
                  loc='lower right', fontsize=fs,
                  )

    ## Sub-solar metallicities ##
    tokeep = np.abs(np.log10(Z/(0.1*Zsun))) < 0.25
    print('Number of BBH mergers in sub-solar clusters: %i' %np.sum(tokeep))
    p = contour_q.get_paths()[0]
    mask_q = p.contains_points(np.hstack((np.log10(m1)[tokeep,np.newaxis],q[tokeep,np.newaxis])))
    p = contour_a.get_paths()[0]
    mask_a = p.contains_points(np.hstack((np.log10(m1)[tokeep,np.newaxis],a1[tokeep,np.newaxis])))
    GW241110 = mask_q & mask_a
    print('Number of GW241110-like mergers:', np.sum(GW241110))
    axes[0,0].scatter(np.log10(m1[tokeep][GW241110]), a1[tokeep][GW241110],
                      c='darkorange', s=5**2,
                      )
    axes[1,0].scatter(np.log10(m1[tokeep][GW241110]), q[tokeep][GW241110],
                      c='darkorange', s=5**2,
                      )
    axes[0,0].set_title(r'$Z\sim\ 0.1\ \mathrm{Z}_\odot$', fontsize=fs)
    """
    tokeep = np.abs(np.log10(Z/(0.1*Zsun))) < 0.25
    axes[0,0].scatter(m1[tokeep], a1[tokeep],
                      c='darkorange', s=5**2,
                      )
    axes[1,0].scatter(m1[tokeep], q[tokeep],
                      c='darkorange', s=5**2,
                      )
    """

    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'Fig6_GW241110-like.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2'
    grid_dir = os.path.join(FIRE_dir,'grid')
    sim_names = ['m12i','m11d','m11h','m11e','m11q','m11i']

    GW24_dir = '/home/tristan-bruel/Documents/Science/o4a-astro/GW241011_GW241110_data_release/data'
    w_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical'
    gw241011_color = '#1f78b4'
    gw241110_color = '#33a02c'

    plot_mBH_spin_q()
    plt.show()
