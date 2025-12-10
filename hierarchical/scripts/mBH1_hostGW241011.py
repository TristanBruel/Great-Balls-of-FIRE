import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

import os


def plot_mBH1():


    #####################################
    # Plot your result
    fs = 10
    lw = 1.4
    plt.rc('font', family='serif')
    plt.rc('lines', linewidth=lw)
    plt.rc('text', usetex=True)

    fig, ax = plt.subplots(1, 1, figsize=(5.,5.))
    
    m1bins = np.linspace(0,200,51)

    ### Show clusters that host GW241011-like BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs.dat')
    cluster_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1
    
    # GW241110 posterior #
    gw241110_qm = np.loadtxt(os.path.join(w_dir,'gw241110_qm.txt'))
    q_bins_241110 = gw241110_qm[1:,0]
    log_m1_bins_241110 = gw241110_qm[0,1:]
    heights_241110_q = gw241110_qm[1:,1:]
    c90_241110_q = [gw241110_qm[0,0]]
    gw241110_am = np.loadtxt(os.path.join(w_dir,'gw241110_am.txt'))
    s1_bins_241110 = gw241110_am[1:,0]
    heights_241110_a = gw241110_am[1:,1:]
    c90_241110_a = [gw241110_am[0,0]]

    # GW241011-like BBH mergers #
    """
    GW241110 = np.zeros(m1.size, dtype='bool')
    for i in range(m1.size):
            m = np.log10(m1)[i]
            mr = q[i]
            a = a1[i]
            if m<log_m1_bins_241110.min() or mr<q_bins_241110.min() or a<s1_bins_241110.min():
                continue
            which_mbin = np.where(m>=log_m1_bins_241110)[0][-1]
            which_qbin = np.where(mr>=q_bins_241110)[0][-1]
            which_abin = np.where(a>=s1_bins_241110)[0][-1]
            if heights_241110_q.reshape(q_bins_241110.size, log_m1_bins_241110.size)[which_qbin,which_mbin]>c90_241110_q:
                if heights_241110_a.reshape(s1_bins_241110.size, log_m1_bins_241110.size)[which_abin,which_mbin]>c90_241110_a:
                    GW241110[i] = 1
    """
    fig2, ax2 = plt.subplots(1, 1, figsize=(3.,3.))
    contour_q = ax2.contour(log_m1_bins_241110, q_bins_241110, heights_241110_q,
                           levels=[*c90_241110_q, np.inf], colors=gw241110_color)
    p = contour_q.get_paths()[0]
    mask_q = p.contains_points(np.hstack((np.log10(m1)[:,np.newaxis],q[:,np.newaxis])))
    contour_a = ax2.contour(log_m1_bins_241110, s1_bins_241110, heights_241110_a,
                           levels=[*c90_241110_a, np.inf], colors=gw241110_color)
    p = contour_a.get_paths()[0]
    mask_a = p.contains_points(np.hstack((np.log10(m1)[:,np.newaxis],a1[:,np.newaxis])))
    GW241110 = mask_q & mask_a
    plt.close(fig2)
    print('Number of GW241110-like BBH mergers:', np.sum(GW241110))
    unique, indexes = np.unique(cluster_id[GW241110], return_index=True)
    print('Number of clusters hosting these events:', len(unique))
    mBH = np.array([])
    for i in range(len(unique)):
        mBH = np.append(mBH,m1[cluster_id==unique[i]])
    print('Total number of BBH mergers from these clusters:',len(mBH),'\n')
    hist, _ = np.histogram(mBH, bins=m1bins)
    ax.stairs(hist, m1bins,
              color='firebrick', label='my sims',
              )
    hist, _ = np.histogram(m1[GW241110], bins=m1bins)
    ax.stairs(hist, m1bins,
              color='firebrick', fill=True,
              )

    ## BBH mergers from Kremer CMC catalog ##
    bbh_file = os.path.join(GW24_dir, 'external_cluster_simulations/BBHmergers.dat')
    cluster_id, rv, Z, N, m1, m2, a1, a2 = np.loadtxt(bbh_file, usecols=(0,1,3,4,8,9,10,11)).T
    Mcl = N *0.6 # mean mass for Kroupa 2001 IMF
    rhm = rv *0.833 # conversion from virial to half-mass radius for king profile (w_0=5)
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1
    print('Total number of BBH mergers in Kremer CMC catalog: {}'.format(len(m1)))
    print('Number of hierarchical BBH mergers: {}'.format(np.sum(a1+a2>0)))
    """
    GW241110 = np.zeros(m1.size, dtype='bool')
    for i in range(m1.size):
            m = np.log10(m1)[i]
            mr = q[i]
            a = a1[i]
            if m<log_m1_bins_241110.min() or mr<q_bins_241110.min() or a<s1_bins_241110.min():
                continue
            which_mbin = np.where(m>=log_m1_bins_241110)[0][-1]
            which_qbin = np.where(mr>=q_bins_241110)[0][-1]
            which_abin = np.where(a>=s1_bins_241110)[0][-1]
            if heights_241110_q.reshape(q_bins_241110.size, log_m1_bins_241110.size)[which_qbin,which_mbin]>c90_241110_q:
                if heights_241110_a.reshape(s1_bins_241110.size, log_m1_bins_241110.size)[which_abin,which_mbin]>c90_241110_a:
                    GW241110[i] = 1
    """
    p = contour_q.get_paths()[0]
    mask_q = p.contains_points(np.hstack((np.log10(m1)[:,np.newaxis],q[:,np.newaxis])))
    p = contour_a.get_paths()[0]
    mask_a = p.contains_points(np.hstack((np.log10(m1)[:,np.newaxis],a1[:,np.newaxis])))
    GW241110 = mask_q & mask_a
    print('Number of GW241110-like mergers:', np.sum(GW241110))
    unique, indexes = np.unique(cluster_id[GW241110], return_index=True)
    print('Number of clusters hosting these events:', len(unique))
    mBH = np.array([])
    for i in range(len(unique)):
        mBH = np.append(mBH,m1[cluster_id==unique[i]])
    print('Total number of BBH mergers from these clusters:',len(mBH),'\n')
    hist, _ = np.histogram(mBH, bins=m1bins)
    ax.stairs(hist, m1bins,
              color='darkorange', label='Kremer',
              )
    hist, _ = np.histogram(m1[GW241110], bins=m1bins)
    ax.stairs(hist, m1bins,
              color='darkorange', fill=True, zorder=0,
              )

    ax.stairs(np.zeros_like(hist), m1bins,
              color='gray', fill=True,
              label='GW241110-like mergers',
              )

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(xmin=0, xmax=170)
    xticks = np.linspace(0,150,7)
    ax.set_xticks(xticks)
    ax.set_ylabel(r'$\mathrm{N}_\mathrm{BBHs}$', fontsize=fs)
    ax.set_ylim(ymin=0.8, ymax=2e3)
    ax.set_yscale('log')
    ax.set_box_aspect(1)
    ax.legend(loc='upper right', fontsize=fs)


    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'mBH1_host_GW241011.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    grid_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIRE-2/grid'

    GW24_dir = '/home/tristan-bruel/Documents/Science/o4a-astro/GW241011_GW241110_data_release/data'
    w_dir = '/home/tristan-bruel/Documents/Science/GBoF/hierarchical'
    gw241011_color = '#1f78b4'
    gw241110_color = '#33a02c'

    plot_mBH1()
    plt.show()
