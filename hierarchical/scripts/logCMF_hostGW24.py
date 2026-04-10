import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
from glob import glob


def plot_CMF_detectable(method='SFR'):

    cluster_file = os.path.join(grid_dir, 'gridClusters_withKremer.hdf5')
    with h5py.File(cluster_file) as c:
        cluster_masses = c['Mass'][()]
        cluster_radii = c['HalfMassRadius'][()]
        cluster_metallicities = c['Metallicity'][()]
        cluster_ids = c['id'][()]
        cluster_nBBHs = c['nBBHs'][()]
    cluster_ids = np.astype(cluster_ids, np.str_)

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
                      '#f781bf', '#a65628', '#e41a1c',
                      '#984ea3', '#999999', '#dede00']

    fig, ax = plt.subplots(1, 1, figsize=(5.,5.))

    logmbins = np.linspace(4.,7.,13)

    bbhfiles = glob(os.path.join(bbhs_dir, 'detectableBBHs_{}_seed*.h5'.format(method)))
    Ncl_mergers = np.zeros((len(bbhfiles),len(logmbins)))
    Ncl_det = np.zeros((len(bbhfiles),len(logmbins)))
    Ncl_GW24 = np.zeros((len(bbhfiles),len(logmbins)))
    Ncl_GW24_det = np.zeros((len(bbhfiles),len(logmbins)))
    for n,bbhfile in enumerate(bbhfiles):
        with h5py.File(bbhfile, 'r') as bbhs:
            m1 = bbhs['m1'][:]
            m2 = bbhs['m2'][:]
            a1 = bbhs['a1'][:]
            pdet = bbhs['pdet'][:,0]
            clid = bbhs['cluster_id'][:]
        clid = np.astype(clid, np.str_)
        #print('Number of BBH mergers: %i' %len(pdet))
        #print('Number of detectable mergers: %i' %np.sum(pdet))
        lochost = np.array([np.where(cluster_ids==u)[0][0] for u in clid])
        Ncl, _ = np.histogram(np.log10(cluster_masses[lochost]), bins=logmbins, weights=pdet)
        Ncl_det[n,:] = np.append(0,Ncl)
        
        # BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses
        GW24 = (m1<30) & (m2/m1<0.85) & (a1>0.05)
        lochost = np.array([np.where(cluster_ids==u)[0][0] for u in clid[GW24]])
        Ncl, _ = np.histogram(np.log10(cluster_masses[lochost]), bins=logmbins, 
                              weights=pdet[GW24],
                              )
        Ncl_GW24_det[n,:] = np.append(0,Ncl)
        

    Ncl_det = np.asarray(Ncl_det)
    Ncl_low = np.percentile(Ncl_det, 10, axis=0)
    Ncl_up = np.percentile(Ncl_det, 90, axis=0)
    Ncl_med = np.percentile(Ncl_det, 50, axis=0)
    ax.plot(logmbins, Ncl_med,
            color=CB_color_cycle[0],
            label='Hosts of detectable BBH mergers',
            )
    ax.fill_between(logmbins, Ncl_low, Ncl_up,
                    alpha=0.4, color=CB_color_cycle[0],
                    )

    Ncl_GW24_det = np.asarray(Ncl_GW24_det)
    Ncl_low = np.percentile(Ncl_GW24_det, 10, axis=0)
    Ncl_up = np.percentile(Ncl_GW24_det, 90, axis=0)
    Ncl_med = np.percentile(Ncl_GW24_det, 50, axis=0)
    ax.plot(logmbins, Ncl_med,
            color=CB_color_cycle[1],
            label='Hosts of detectable \n GW24-like mergers',
            )
    ax.fill_between(logmbins, Ncl_low, Ncl_up,
                    alpha=0.4, color=CB_color_cycle[1],
                    )

    ax.set_xlabel(r'$\mathrm{log}_{10}(M_\mathrm{cl}/\mathrm{M}_\odot)$')
    ax.set_xlim(xmin=4.5, xmax=7)
    ax.set_xticks(np.linspace(4.5,7,6))
    ax.set_ylabel(r'$\mathrm{N}_\mathrm{cl}$')
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-2, ymax=1e2)
    ax.set_box_aspect(1)
    ax.legend(loc='upper left')



    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'CMF_hosts_detectableBBHs.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'
    bbhs_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/BBHs'

    plot_CMF_detectable()
    plt.show()
