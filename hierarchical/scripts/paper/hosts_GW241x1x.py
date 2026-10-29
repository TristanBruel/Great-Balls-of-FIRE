import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

from plotting_utils import get_contour_heights, Bounded_2d_kde

import os
import argparse



def plot_clusters_Mcl_rhm(events, Kremer=False, nospin=False, plot_dir=None):

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
    color_logB = CB_color_cycle[0] # blue
    color_logB_astro = CB_color_cycle[7] # red

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
    cl_id_bbhs = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    m1, m2 = np.maximum(m1, m2), np.minimum(a1,a2)
    hierar = a1+a2>0
    unique, indexes = np.unique(cl_id_bbhs[hierar], return_index=True)
    host_hierar = np.array([np.where(cluster_id==u)[0][0] for u in unique])

    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/scripts/paper/log_Bayes_factors'
    logm_bins = np.linspace(4, 7, 31)
    midm_bins = (logm_bins[1:]+logm_bins[:-1])/2
    logr_bins = np.linspace(-3, 3, 61)
    midr_bins = (logr_bins[1:]+logr_bins[:-1])/2
    for event in events:
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

        ### Load the computed log-B factors ###
        filename = 'logBF_%s_PE' %event + nospin*'_nospin' + '.dat'
        logBF_file = os.path.join(w_dir, filename)
        cl_id_logBF = np.loadtxt(logBF_file, skiprows=1, usecols=0, dtype=str)
        logBF = np.loadtxt(logBF_file, skiprows=1, usecols=1)
        loc_cl = np.array([np.where(cluster_id==clid)[0][0] for clid in cl_id_logBF])
        H_logBF, xedges, yedges = np.histogram2d(
                x=logMcl[loc_cl],
                y=logRhm[loc_cl],
                bins=[logm_bins,logr_bins],
                weights=np.exp(logBF),
                density=True,
        )
        tokeep = logBF > logBF.max()-3
        loc_top_cl = np.array([np.where(cluster_id==clid)[0][0] for clid in cl_id_logBF[tokeep]])
        ax.scatter(logMcl[loc_top_cl], logRhm[loc_top_cl],
                   marker='o', s=7**2, linewidths=1.5,
                   facecolor='none', edgecolor=color_logB,
                   zorder=2,
                   #label=r'Highest $\mathrm{log}\mathcal{B}$'
                   )

        ### Load the computed log-B factors with astro priors ###
        filename = 'logBF_%s_PE_astro' %event + nospin*'_nospin' + '.dat'
        logBF_file = os.path.join(w_dir, filename)
        cl_id_logBF = np.loadtxt(logBF_file, skiprows=1, usecols=0, dtype=str)
        logBF = np.loadtxt(logBF_file, skiprows=1, usecols=1)
        loc_cl = np.array([np.where(cluster_id==clid)[0][0] for clid in cl_id_logBF])
        H_logBF_astro, xedges, yedges = np.histogram2d(
                x=logMcl[loc_cl],
                y=logRhm[loc_cl],
                bins=[logm_bins,logr_bins],
                weights=np.exp(logBF),
                density=True,
        )
        tokeep = logBF > logBF.max()-3
        loc_top_cl = np.array([np.where(cluster_id==clid)[0][0] for clid in cl_id_logBF[tokeep]])
        ax.scatter(logMcl[loc_top_cl], logRhm[loc_top_cl],
                   marker='o', s=6**2, linewidths=1.5,
                   facecolor=color_logB_astro, edgecolor='none',
                   zorder=2,
                   #label=r'Highest $\mathrm{log}\mathcal{B}$'
                   #     +'\n with astro priors'
                   )

        ## Mass marginal on the top subplot ##
        ax_top = ax.inset_axes([0, 1.05, 1, 0.4])
        CMF = np.sum(H /np.diff(logr_bins), axis=1)
        CMF /= np.sum(CMF *np.diff(logm_bins))
        ax_top.plot(midm_bins, CMF,
                    c=color_all,
                    )
        CMF_hierar = np.sum(H_hierar /np.diff(logr_bins), axis=1)
        CMF_hierar /= np.sum(CMF_hierar *np.diff(logm_bins))
        ax_top.plot(midm_bins, CMF_hierar,
                    c=color_hierar,
                    )
        CMF_logBF = np.sum(H_logBF /np.diff(logr_bins), axis=1)
        CMF_logBF /= np.sum(CMF_logBF *np.diff(logm_bins))
        ax_top.plot(midm_bins, CMF_logBF,
                    c=color_logB,
                    )
        CMF_logBF_astro = np.sum(H_logBF_astro /np.diff(logr_bins), axis=1)
        CMF_logBF_astro /= np.sum(CMF_logBF_astro *np.diff(logm_bins))
        ax_top.plot(midm_bins, CMF_logBF_astro,
                    c=color_logB_astro,
                    )

        ## Hal-mass radius marginal on the left subplot ##
        ax_left = ax.inset_axes([-.45, 0, 0.4, 1], sharey=ax)
        CRF = np.sum(H /np.diff(logm_bins)[:,None], axis=0)
        CRF /= np.sum(CRF *np.diff(logr_bins))
        ax_left.plot(CRF, midr_bins,
                     c=color_all,
                     )
        CRF_hierar = np.sum(H_hierar /np.diff(logm_bins)[:,None], axis=0)
        CRF_hierar /= np.sum(CRF_hierar *np.diff(logr_bins))
        ax_left.plot(CRF_hierar, midr_bins,
                     c=color_hierar,
                     )
        CRF_logBF = np.sum(H_logBF /np.diff(logm_bins)[:,None], axis=0)
        CRF_logBF /= np.sum(CRF_logBF *np.diff(logr_bins))
        ax_left.plot(CRF_logBF, midr_bins,
                    c=color_logB,
                    )
        CRF_logBF_astro = np.sum(H_logBF_astro /np.diff(logm_bins)[:,None], axis=0)
        CRF_logBF_astro /= np.sum(CRF_logBF_astro *np.diff(logr_bins))
        ax_left.plot(CRF_logBF_astro, midr_bins,
                    c=color_logB_astro,
                    )

        ax_left.set_xlabel(r'$\mathrm{dn}/\mathrm{d}r_\mathrm{hm}\ [\mathrm{pc}{-1}]$')
        ax_left.set_xscale('log')
        ax_left.set_xlim(xmin=1e-2, xmax=1e1)
        ax_left.set_xticks(np.logspace(-2,1,4))
        ax_left.set_xticks([], minor=True)
        ax_left.set_ylabel(r'$\mathrm{log}(r_\mathrm{hm}/\mathrm{pc})$')
        ax_left.set_ylim(ymin=-3., ymax=3.)
        ax_left.set_yticks(np.linspace(-3,3,7))

        # Add redshift axis on top
        ax_top.set_xlim(xmin=4.5, xmax=7)
        ax_top.set_xticks([])

        ax_top.set_ylabel(r'$\mathrm{dn}/\mathrm{d}M_\mathrm{cl}\ [\mathrm{M}_\odot^{-1}]$')
        ax_top.set_yscale('log')
        ax_top.set_ylim(ymin=1e-2, ymax=1e1)
        ax_top.set_yticks(np.logspace(-2,1,4))
        ax_top.set_yticks([], minor=True)
        #ax3 = ax_top.twinx()
        #ax3.set_ylabel(r'$\mathrm{dn}/\mathrm{d}M_\mathrm{cl}\ [\mathrm{M}_\odot^{-1}]$')
        #ax3.set_yscale('log')
        #ax3.set_ylim(ymin=1e-2, ymax=1e1)
        #ax3.set_yticks(np.logspace(-2,1,4))
        #ax3.set_yticks([], minor=True)

        #ax_top.legend(bbox_to_anchor=(0.,-0.1), loc='lower right')

        ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$')
        ax.set_xlim(xmin=4.5, xmax=7)
        ax.set_xticks(np.linspace(5,7,3))
        ax.set_ylim(ymin=-3., ymax=3.)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.legend(loc='lower right')
        ax.set_box_aspect(1)

        # Save
        plt.savefig(os.path.join(plot_dir,'host_%s_Mcl_rhm' %event +nospin*'_nospin' +'.pdf'), format='pdf', bbox_inches='tight')


def plot_clusters_Mcl_Z(events, Kremer=False, nospin=False, plot_dir=None):

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
    color_logB = CB_color_cycle[0] # blue
    color_logB_astro = CB_color_cycle[7] # red


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
    logZ = np.log10(cluster_Z)

    ### Identify clusters that host hierarchical BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs'+Kremer*'_withKremer'+'.dat')
    cl_id_bbhs = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    m1, m2 = np.maximum(m1, m2), np.minimum(a1,a2)
    hierar = a1+a2>0
    unique, indexes = np.unique(cl_id_bbhs[hierar], return_index=True)
    host_hierar = np.array([np.where(cluster_id==u)[0][0] for u in unique])

    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/scripts/paper/log_Bayes_factors'
    logm_bins = np.linspace(4, 7, 31)
    midm_bins = (logm_bins[1:]+logm_bins[:-1])/2
    logZ_bins = np.linspace(-3, 1, 41)
    midZ_bins = (logZ_bins[1:]+logZ_bins[:-1])/2
    for event in events:
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
                   #label='Hosts of hierarchical mergers'
                   )

        ### Load the computed log-B factors ###
        filename = 'logBF_%s_PE' %event + nospin*'_nospin' + '.dat'
        logBF_file = os.path.join(w_dir, filename)
        cl_id_logBF = np.loadtxt(logBF_file, skiprows=1, usecols=0, dtype=str)
        logBF = np.loadtxt(logBF_file, skiprows=1, usecols=1)
        loc_cl = np.array([np.where(cluster_id==clid)[0][0] for clid in cl_id_logBF])
        H_logBF, xedges, yedges = np.histogram2d(
                x=logMcl[loc_cl], 
                y=logZ[loc_cl], 
                bins=[logm_bins,logZ_bins],
                weights=np.exp(logBF),
                density=True,
        )
        tokeep = logBF > logBF.max()-3
        print('Cluster ids with the highest logB (std):',cl_id_logBF[tokeep])
        loc_top_cl = np.array([np.where(cluster_id==clid)[0][0] for clid in cl_id_logBF[tokeep]])
        ax.scatter(logMcl[loc_top_cl], logZ[loc_top_cl], 
                   marker='o', s=7**2, linewidths=1.5,
                   facecolor='none', edgecolor=color_logB,
                   zorder=2,
                   #label=r'Highest $\mathcal{B}$ (std LVK)'
                   label=r'$\mathrm{log}(\mathcal{B}/\mathcal{B}_\mathrm{max})>-3$ (std LVK)'
                   )

        ### Load the computed log-B factors with astro priors ###
        filename = 'logBF_%s_PE_astro' %event + nospin*'_nospin' + '.dat'
        logBF_file = os.path.join(w_dir, filename)
        cl_id_logBF = np.loadtxt(logBF_file, skiprows=1, usecols=0, dtype=str)
        logBF = np.loadtxt(logBF_file, skiprows=1, usecols=1)
        loc_cl = np.array([np.where(cluster_id==clid)[0][0] for clid in cl_id_logBF])
        H_logBF_astro, xedges, yedges = np.histogram2d(
                x=logMcl[loc_cl],
                y=logZ[loc_cl],
                bins=[logm_bins,logZ_bins],
                weights=np.exp(logBF),
                density=True,
        )
        tokeep = logBF > logBF.max()-3
        print('Cluster ids with the highest logB (astro):',cl_id_logBF[tokeep])
        loc_top_cl = np.array([np.where(cluster_id==clid)[0][0] for clid in cl_id_logBF[tokeep]])
        ax.scatter(logMcl[loc_top_cl], logZ[loc_top_cl], 
                   marker='o', s=6**2, linewidths=1.5,
                   facecolor=color_logB_astro, edgecolor='none',
                   zorder=2,
                   #label=r'Highest $\mathcal{B}$ (astro)'
                   label=r'$\mathrm{log}(\mathcal{B}/\mathcal{B}_\mathrm{max})>-3$ (astro)'
                   )

        
        ## Mass marginal on the top subplot ##
        ax_top = ax.inset_axes([0, 1.05, 1, 0.4])
        CMF = np.sum(H /np.diff(logZ_bins), axis=1)
        CMF /= np.sum(CMF *np.diff(logm_bins))
        ax_top.plot(midm_bins, CMF,
                    c=color_all, 
                    )
        CMF_hierar = np.sum(H_hierar /np.diff(logZ_bins), axis=1)
        CMF_hierar /= np.sum(CMF_hierar *np.diff(logm_bins))
        ax_top.plot(midm_bins, CMF_hierar,
                    c=color_hierar, 
                    )
        CMF_logBF = np.sum(H_logBF /np.diff(logZ_bins), axis=1)
        CMF_logBF /= np.sum(CMF_logBF *np.diff(logm_bins))
        ax_top.plot(midm_bins, CMF_logBF,
                    c=color_logB, 
                    )
        CMF_logBF_astro = np.sum(H_logBF_astro /np.diff(logZ_bins), axis=1)
        CMF_logBF_astro /= np.sum(CMF_logBF_astro *np.diff(logm_bins))
        ax_top.plot(midm_bins, CMF_logBF_astro,
                    c=color_logB_astro, 
                    )

        ## Metallicity marginal on the left subplot ##
        ax_left = ax.inset_axes([-.45, 0, 0.4, 1], sharey=ax)
        CRF = np.sum(H /np.diff(logm_bins)[:,None], axis=0)
        CRF /= np.sum(CRF *np.diff(logZ_bins))
        ax_left.plot(CRF, midZ_bins,
                     c=color_all,
                     )
        CRF_hierar = np.sum(H_hierar /np.diff(logm_bins)[:,None], axis=0)
        CRF_hierar /= np.sum(CRF_hierar *np.diff(logZ_bins))
        ax_left.plot(CRF_hierar, midZ_bins,
                     c=color_hierar,
                     )
        CRF_logBF = np.sum(H_logBF /np.diff(logm_bins)[:,None], axis=0)
        CRF_logBF /= np.sum(CRF_logBF *np.diff(logZ_bins))
        ax_left.plot(CRF_logBF, midZ_bins,
                    c=color_logB, 
                    )
        CRF_logBF_astro = np.sum(H_logBF_astro /np.diff(logm_bins)[:,None], axis=0)
        CRF_logBF_astro /= np.sum(CRF_logBF_astro *np.diff(logZ_bins))
        ax_left.plot(CRF_logBF_astro, midZ_bins,
                    c=color_logB_astro, 
                    )

        ax_left.set_xlabel(r'$\mathrm{dn}/\mathrm{d}Z\ [\mathrm{Z}_\odot^{-1}]$')
        ax_left.set_xscale('log')
        ax_left.set_xlim(xmin=1e-2, xmax=1e1)
        ax_left.set_xticks(np.logspace(-2,1,4))
        ax_left.set_xticks([], minor=True)
        ax_left.set_ylabel(r'$\mathrm{log}(Z/\mathrm{Z}_\odot)$')
        ax_left.set_ylim(ymin=-3., ymax=1.)
        ax_left.set_yticks(np.linspace(-3,1,5))
        
        # Add redshift axis on top
        ax_top.set_xlim(xmin=4.5, xmax=7)
        ax_top.set_xticks([])

        ax_top.set_ylabel(r'$\mathrm{dn}/\mathrm{d}M_\mathrm{cl}\ [\mathrm{M}_\odot^{-1}]$')
        ax_top.set_yscale('log')
        ax_top.set_ylim(ymin=1e-2, ymax=1e1)
        ax_top.set_yticks(np.logspace(-2,1,4))
        ax_top.set_yticks([], minor=True)
        #ax3 = ax_top.twinx()
        #ax3.set_ylabel(r'$\mathrm{dn}/\mathrm{d}M_\mathrm{cl}\ [\mathrm{M}_\odot^{-1}]$')
        #ax3.set_yscale('log')
        #ax3.set_ylim(ymin=1e-2, ymax=1e1)
        #ax3.set_yticks(np.logspace(-2,1,4))
        #ax3.set_yticks([], minor=True)

        #ax_top.legend(bbox_to_anchor=(0.,-0.1), loc='lower right')

        ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$')
        ax.set_xlim(xmin=4.5, xmax=7)
        ax.set_xticks(np.linspace(5,7,3))
        ax.set_ylim(ymin=-3., ymax=1.)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.legend(loc='lower right', title=event)
        ax.set_box_aspect(1)

        # Save
        plt.savefig(os.path.join(plot_dir,'host_%s_Mcl_Z' %event +nospin*'_nospin' +'.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots/paper'

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", help="GW event to consider", nargs='+', default=['GW241011','GW241110'])
    parser.add_argument("--Kremer", action='store_true', help="Include CMC clusters from Kremer et al. 2018")
    parser.add_argument("--nospin", dest='nospin', action='store_true', help="Add results with logB computed without considering spins")
    args = parser.parse_args()

    plot_clusters_Mcl_Z(events=args.events, 
                        Kremer=args.Kremer, nospin=args.nospin, 
                        plot_dir=plot_dir,
                        )
    plot_clusters_Mcl_rhm(events=args.events, 
                        Kremer=args.Kremer, nospin=args.nospin, 
                        plot_dir=plot_dir,
                        )
    plt.show()
