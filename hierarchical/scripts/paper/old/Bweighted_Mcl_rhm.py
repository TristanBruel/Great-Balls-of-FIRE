import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import argparse



def plot_clusters_Mcl_rhm(astro=False, nospin=False, Kremer=False, plot_dir=None):

    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'
    cluster_file = os.path.join(grid_dir, 'gridClusters'+Kremer*'_withKremer'+'.hdf5')
    with h5py.File(cluster_file) as c:
        cluster_id = c['id'][()]
        cluster_mass = c['Mass'][()]
        cluster_radius = c['HalfMassRadius'][()]
        cluster_Z = c['Metallicity'][()]
        cluster_nBBHs = c['nBBHs'][()]
    cluster_id = np.astype(cluster_id, str)
    print('Number of CMC clusters:', len(cluster_id))

    ### Identify clusters that host hierarchical BBH mergers ###
    bbhfile = os.path.join(grid_dir, 'allBBHs'+Kremer*'_withKremer'+'.dat')
    cl_id_bbhs = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    m1, m2 = np.maximum(m1, m2), np.minimum(a1,a2)
    hierar = a1+a2>0
    print('Number of hierarchical mergers in all cluster simulations:', np.sum(hierar))
    unique, indexes = np.unique(cl_id_bbhs[hierar], return_index=True)
    host_hierar = np.array([np.where(cluster_id==u)[0][0] for u in unique])
    print('Number of CMC clusters with hierarchical mergers:', len(host_hierar))

    ### Load the computed log-B factors ###
    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical'
    for event in ['GW241011','GW241110']:
        outfile = 'logBF_%s_PE' %event + astro*'_astro' + '.dat'
        logBF_file = os.path.join(w_dir, outfile)
        cl_id_logBF = np.loadtxt(logBF_file, skiprows=1, usecols=0, dtype=str)
        logBF = np.loadtxt(logBF_file, skiprows=1, usecols=1)
        weights = np.zeros(len(cluster_mass))
        for n,c in enumerate(cl_id_logBF):
            ind = np.where(cluster_id==c)[0]
            weights[ind] = 10**logBF[n]

        if nospin:
            outfile = 'logBF_%s_PE' %event + astro*'_astro' + '_nospin.dat'
            logBF_file = os.path.join(w_dir, outfile)
            cl_id_logBF_nospin = np.loadtxt(logBF_file, skiprows=1, usecols=0, dtype=str)
            logBF_nospin = np.loadtxt(logBF_file, skiprows=1, usecols=1)
            weights_nospin = np.zeros(len(cluster_mass))
            for n,c in enumerate(cl_id_logBF_nospin):
                ind = np.where(cluster_id==c)[0]
                weights_nospin[ind] = 10**logBF_nospin[n]

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
        
        fig, ax = plt.subplots(1, 1, figsize=(4.5,4.5))

        logm_bins = np.linspace(4, 7, 31)
        midm_bins = (logm_bins[1:]+logm_bins[:-1])/2
        logr_bins = np.linspace(-3, 3, 61)
        midr_bins = (logr_bins[1:]+logr_bins[:-1])/2

        H, xedges, yedges = np.histogram2d(x=np.log10(cluster_mass), y=np.log10(cluster_radius), 
                                           bins=[logm_bins,logr_bins],
                                           )
        H_hierar, xedges, yedges = np.histogram2d(x=np.log10(cluster_mass[host_hierar]), 
                                                  y=np.log10(cluster_radius[host_hierar]), 
                                                  bins=[logm_bins,logr_bins],
                                                  )
        HB, xedges, yedges = np.histogram2d(x=np.log10(cluster_mass), y=np.log10(cluster_radius), 
                                           bins=[logm_bins,logr_bins],
                                           weights=weights,
                                           )
        if nospin:
            HB_nospin, xedges, yedges = np.histogram2d(x=np.log10(cluster_mass), y=np.log10(cluster_radius),
                                               bins=[logm_bins,logr_bins],
                                               weights=weights_nospin,
                                               )
            HB_nospin /= np.diff(logm_bins)[:,None] *np.diff(logr_bins) *np.sum(HB_nospin)

        H /= np.diff(logm_bins)[:,None] *np.diff(logr_bins) *np.sum(H)
        H_hierar /= np.diff(logm_bins)[:,None] *np.diff(logr_bins) *np.sum(H_hierar)
        HB /= np.diff(logm_bins)[:,None] *np.diff(logr_bins) *np.sum(HB)
        logvmin = (int(10*np.log10(H[H>0].min()))-1) /10
        logvmax = (int(10*np.log10(H[H>0].max()))+1) /10
        H[H<1e-32] = 1e-32

        cmap = plt.get_cmap('plasma')
        vmin = 10**logvmin
        vmax = 10**logvmax
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

        X, Y = np.meshgrid(midm_bins, midr_bins)
        im = ax.contourf(X, Y, H.T,
                         levels=np.logspace(logvmin,logvmax,10),
                         cmap=cmap,
                         norm=norm,
                        )
        ax.scatter(np.log10(cluster_mass[host_hierar]), 
                   np.log10(cluster_radius[host_hierar]),
                   marker='d', c='forestgreen', s=5**2,
                   )

        ax_top = ax.inset_axes([0, 1.05, 1, 0.4])
        ax_left = ax.inset_axes([-.45, 0, 0.4, 1], sharey=ax)
        
        CMF = np.sum(H *np.diff(logr_bins), axis=1)
        ax_top.plot(midm_bins, CMF,
                    c='k', label='all MSCs',
                    )
        CMF_hierar = np.sum(H_hierar *np.diff(logr_bins), axis=1)
        ax_top.plot(midm_bins, CMF_hierar,
                    c='forestgreen', label='Hosts of\nhierarchical mergers',
                    )
        CMF_HB = np.sum(HB *np.diff(logr_bins), axis=1)
        ax_top.plot(midm_bins, CMF_HB,
                    c='firebrick', label=r'$\mathcal{B}$-weighted '+event,
                    )
        if nospin:
            CMF_HB_nospin = np.sum(HB_nospin *np.diff(logr_bins), axis=1)
            ax_top.plot(midm_bins, CMF_HB_nospin,
                        c='royalblue', label=r'$\mathcal{B}$-weighted '+event+'\nwithout spin',
                        )

        CRF = np.sum(H *np.diff(logm_bins)[:,None], axis=0)
        ax_left.plot(CRF, midr_bins,
                     c='k',
                     )
        CRF_hierar = np.sum(H_hierar *np.diff(logm_bins)[:,None], axis=0)
        ax_left.plot(CRF_hierar, midr_bins,
                     c='forestgreen',
                     )
        CRF_HB = np.sum(HB *np.diff(logm_bins)[:,None], axis=0)
        ax_left.plot(CRF_HB, midr_bins,
                     c='firebrick',
                     )
        if nospin:
            CRF_HB_nospin = np.sum(HB_nospin *np.diff(logm_bins)[:,None], axis=0)
            ax_left.plot(CRF_HB_nospin, midr_bins,
                         c='royalblue',
                         )

        ax_left.set_xlabel(r'$\mathrm{dn}/\mathrm{d}r_\mathrm{hm}\ [\mathrm{pc}^{-1}]$')
        ax_left.set_xscale('log')
        ax_left.set_xlim(xmin=1e-4, xmax=1e2)
        ax_left.set_xticks(np.logspace(-4,2,4))
        ax_left.set_ylabel(r'$\mathrm{log}(r_\mathrm{hm}/\mathrm{pc})$')
        ax_left.set_ylim(ymin=-2.5, ymax=2.5)
        ax_left.set_yticks(np.linspace(-2,2,5))
        
        # Add redshift axis on top
        ax_top.set_xlim(xmin=4.5, xmax=7)
        ax_top.set_xticks([])

        ax_top.set_yscale('log')
        ax_top.set_ylim(ymin=1e-4, ymax=1e2)
        ax_top.set_yticks([])
        ax_top.set_yticks([], minor=True)
        ax3 = ax_top.twinx()
        ax3.set_ylabel(r'$\mathrm{dn}/\mathrm{d}M_\mathrm{cl}\ [\mathrm{M}_\odot^{-1}]$')
        ax3.set_yscale('log')
        ax3.set_ylim(ymin=1e-4, ymax=1e2)
        ax3.set_yticks(np.logspace(-4,2,4))
        ax3.set_yticks([], minor=True)

        ax_top.legend(bbox_to_anchor=(0.,-0.1), loc='lower right')

        ax.set_xlabel(r'$\mathrm{log}(M_\mathrm{cl}/\mathrm{M}_\odot)$')
        ax.set_xlim(xmin=4.5, xmax=7)
        ax.set_xticks(np.linspace(5,7,3))
        ax.set_ylim(ymin=-2.5, ymax=2.5)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.set_box_aspect(1)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.06)
        cbar.set_label(r'$\mathrm{dn}/\mathrm{dlog}(M_\mathrm{cl}/\mathrm{M}_\odot)\mathrm{dlog}(r_\mathrm{hm}/\mathrm{pc})$')
        cbar.ax.tick_params(labelsize=0.8*fs)
      
        # Save
        plt.savefig(os.path.join(plot_dir,'Bweighted_%s'%event+astro*'_astro'+'_Mcl_rhm.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical'
    plot_dir = os.path.join(w_dir,'plots/paper')

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--astro", dest='astro_prior', action='store_true', help="Uses PE samples with astro priors")
    parser.add_argument("--nospin", dest='nospin', action='store_true', help="Add results with logB computed without considering spins")
    parser.add_argument("--Kremer", action='store_true', help="Include CMC clusters from Kremer et al. 2018")
    args = parser.parse_args()

    plot_clusters_Mcl_rhm(astro=args.astro_prior, nospin=args.nospin, Kremer=args.Kremer, plot_dir=plot_dir)
    plt.show()
