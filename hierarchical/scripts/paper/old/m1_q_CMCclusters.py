import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os



def plot_clusters_m1_q(Kremer=True, plot_dir=None):

    ### Identify hierarchical BBH mergers ###
    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'
    bbhfile = os.path.join(grid_dir, 'allBBHs'+Kremer*'_withKremer'+'.dat')
    cl_id_bbhs = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=str)
    m1, m2, a1, a2 = np.loadtxt(bbhfile, usecols=[4,5,8,9], skiprows=1).T
    m1, m2 = np.maximum(m1, m2), np.minimum(m1,m2)
    q = m2 /m1
    print('Number of BBH mergers in all cluster simulations:', len(m1))
    hierar = a1+a2>0
    print('Number of hierarchical mergers in all cluster simulations:', np.sum(hierar))
    unique, indexes = np.unique(cl_id_bbhs[hierar], return_index=True)
    print('Number of CMC clusters with hierarchical mergers:', len(unique))

    ### Identify `low'-mass hierarchical BBH mergers ###
    hierar_lowmass = hierar & (m1<30)
    print('Number of m1<30 hierarchical mergers in all cluster simulations:', np.sum(hierar_lowmass))
    unique, indexes = np.unique(cl_id_bbhs[hierar_lowmass], return_index=True)
    print('Number of CMC clusters with m1<30 hierarchical mergers:', len(unique))

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

    m1_bins = np.linspace(0, 100, 51)
    midm_bins = (m1_bins[1:]+m1_bins[:-1])/2
    q_bins = np.linspace(0, 1, 51)
    midq_bins = (q_bins[1:]+q_bins[:-1])/2

    H, xedges, yedges = np.histogram2d(x=m1, y=q, 
                                       bins=[m1_bins,q_bins],
                                       )
    H_hierar, xedges, yedges = np.histogram2d(x=m1[hierar], 
                                              y=q[hierar], 
                                              bins=[m1_bins,q_bins],
                                              )
    H_lowmass, xedges, yedges = np.histogram2d(x=m1[hierar_lowmass], 
                                               y=q[hierar_lowmass], 
                                               bins=[m1_bins,q_bins],
                                               )
    H /= np.diff(m1_bins)[:,None] *np.diff(q_bins) *np.sum(H)
    H_hierar /= np.diff(m1_bins)[:,None] *np.diff(q_bins) *np.sum(H_hierar)
    H_lowmass /= np.diff(m1_bins)[:,None] *np.diff(q_bins) *np.sum(H_lowmass)
    #HB /= np.diff(logm_bins)[:,None] *np.diff(logr_bins) 
    logvmin = (int(10*np.log10(H[H>0].min()))-1) /10
    logvmax = (int(10*np.log10(H[H>0].max()))+1) /10
    H[H<1e-32] = 1e-32

    cmap = plt.get_cmap('plasma')
    vmin = 10**logvmin
    vmax = 10**logvmax
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    X, Y = np.meshgrid(midm_bins, midq_bins)
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(logvmin,logvmax,10),
                     cmap=cmap,
                     norm=norm,
                    )

    ax_top = ax.inset_axes([0, 1.05, 1, 0.4])
    ax_left = ax.inset_axes([-.45, 0, 0.4, 1], sharey=ax)
    
    CMF = np.sum(H /np.diff(q_bins), axis=1)
    print(np.trapezoid(x=midm_bins, y=CMF))
    ax_top.plot(midm_bins, CMF,
                c='k', label='all MSCs',
                )
    CMF_hierar = np.sum(H_hierar /np.diff(q_bins), axis=1)
    ax_top.plot(midm_bins, CMF_hierar,
                c='forestgreen', label='Hierarchical mergers',
                )
    CMF_hierar_lowmass = np.sum(H_lowmass /np.diff(q_bins), axis=1)
    print(np.trapezoid(x=midm_bins, y=CMF_hierar_lowmass))
    ax_top.plot(midm_bins, CMF_hierar_lowmass,
                c='firebrick', label="`Low'-mass hierarchical mergers",
                )

    CRF = np.sum(H /np.diff(m1_bins)[:,None], axis=0)
    print(np.trapezoid(x=midq_bins, y=CRF))
    ax_left.plot(CRF, midq_bins,
                 c='k',
                 )
    CRF_hierar = np.sum(H_hierar /np.diff(m1_bins)[:,None], axis=0)
    print(np.trapezoid(x=midq_bins, y=CRF_hierar))
    ax_left.plot(CRF_hierar, midq_bins,
                 c='forestgreen',
                 )
    CRF_hierar_lowmass = np.sum(H_lowmass /np.diff(m1_bins)[:,None], axis=0)
    ax_left.plot(CRF_hierar_lowmass, midq_bins,
                 c='firebrick',
                 )

    ax_left.set_xlabel(r'$\mathrm{dn}/\mathrm{d}q$')
    ax_left.set_xscale('log')
    ax_left.set_xlim(xmin=1e-3, xmax=1e3)
    #ax_left.set_xticks(np.logspace(3,6,4))
    ax_left.set_ylabel(r'$q$')
    ax_left.set_ylim(ymin=0, ymax=1.)
    #ax_left.set_yticks(np.linspace(-2,2,5))
    
    # Add redshift axis on top
    ax_top.set_xlim(xmin=1., xmax=101)
    ax_top.set_xticks([])

    ax_top.set_yscale('log')
    ax_top.set_ylim(ymin=1e-3, ymax=1e3)
    ax_top.set_yticks([])
    ax_top.set_yticks([], minor=True)
    ax3 = ax_top.twinx()
    ax3.set_ylabel(r'$\mathrm{dn}/\mathrm{d}m_1\ [\mathrm{M}_\odot^{-1}]$')
    ax3.set_yscale('log')
    ax3.set_ylim(ymin=1e-3, ymax=1e3)
    #ax3.set_yticks(np.logspace(3,6,4))
    ax3.set_yticks([], minor=True)

    ax_top.legend(bbox_to_anchor=(0.,-0.1), loc='lower right')

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$')
    ax.set_xlim(xmin=1., xmax=101)
    #ax.set_xticks(np.linspace(5,7,3))
    ax.set_ylim(ymin=0., ymax=1.)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_box_aspect(1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.06)
    cbar.set_label(r'$\mathrm{dn}/\mathrm{d}(m_1/\mathrm{M}_\odot)\mathrm{d}q$')
    cbar.ax.tick_params(labelsize=0.8*fs)
  
    # Save
    plt.savefig(os.path.join(plot_dir,'m1_q_CMC.pdf'), format='pdf', bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical'
    plot_dir = os.path.join(w_dir,'plots/paper')
    Kremer = False

    plot_clusters_m1_q(Kremer=Kremer, plot_dir=plot_dir)
    plt.show()
