import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

import os


# Function to compute and evaluate KDE over m1 and q space
def eval_kde_m1_q(m1, q, nbins=(100, 99), probs=(0.95, 0.50)):

    """
    Function to evaluate gaussian KDE over posterior samples, with symmetric
    boundary condition at q=1

    Parameters
    ----------
    m1 : array
        Set of m1 PE samples, shape (n_samples,)
    q : array
        Set of q PE samples, shape (n_samples,)

    Returns
    -------
    heights : array
        2D grid of KDE values, evaluated across M and Q. Shape (m, n)
    c90 : float
        Contour level bounding central 90% posterior probability region
    """

    # Create 2D KDE over provided samples
    kde = gaussian_kde(np.array([m1, q]))

    m1_bins = np.linspace(np.min(m1) - 0.1, np.max(m1) + 0.1, nbins[0])
    q_bins = np.linspace(np.min(q) - 0.1, 1, nbins[1])
    M, Q = np.meshgrid(m1_bins, q_bins)

    # Evaluate KDE over gridded M and Q values, imposing symmetric boundary
    heights = kde(np.array([M.reshape(-1), Q.reshape(-1)])) \
            + kde(np.array([M.reshape(-1), 2 - Q.reshape(-1)]))

    # Compute contour level enclosing highest 90% credible probability region
    # First, reshape `heights` into a 1D array of decreasing values
    heights_1d = np.sort(heights.reshape(-1))[::-1]

    # Next, find the height within which lies 95% of the posterior probability
    cs = [np.interp(p, np.cumsum(heights_1d)/np.sum(heights_1d), heights_1d) for p in probs]

    return m1_bins, q_bins, heights, cs

# Function to compute and evaluate KDE over m1 and s1 space
def eval_kde_m1_s1(m1, s1, nbins=(100, 99), probs=(0.95, 0.50)):

    """
    Function to evaluate gaussian KDE over posterior samples, with symmetric
    boundary conditions at s1=0 and s1=0.99

    Parameters
    ----------
    m1 : array
        Set of m1 PE samples, shape (n_samples,)
    s1 : array
        Set of s1 PE samples, shape (n_samples,)

    Returns
    -------
    heights : array
        2D grid of KDE values, evaluated across M and S. Shape (m, n)
    c90 : float
        Contour level bounding central 90% posterior probability region
    """

    # Create 2D KDE over provided samples
    kde = gaussian_kde(np.array([m1, s1]))

    m1_bins = np.linspace(np.min(m1) - 0.1, np.max(m1) + 0.1, nbins[0])
    s1_bins = np.linspace(0, 1, nbins[1])
    M, S = np.meshgrid(m1_bins, s1_bins)

    # Evaluate KDE over gridded M and S values, imposing symmetric boundaries in S
    heights = kde(np.array([M.reshape(-1), S.reshape(-1)])) \
            + kde(np.array([M.reshape(-1), 2*0.99 - S.reshape(-1)])) \
            + kde(np.array([M.reshape(-1), -S.reshape(-1)]))

    # Compute contour level enclosing highest 90% credible probability region
    # First, reshape `heights` into a 1D array of decreasing values
    heights_1d = np.sort(heights.reshape(-1))[::-1]

    # Next, find the height within which lies 95% of the posterior probability
    cs = [np.interp(p, np.cumsum(heights_1d)/np.sum(heights_1d), heights_1d) for p in probs]

    return m1_bins, s1_bins, heights, cs


def plot_mBH_spin(Zsun=0.017, n=1000):

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

    ### First-generation mergers ###
    m1g = m1[a1+a2==0]
    q1g = q[a1+a2==0]
    Z1g = Z[a1+a2==0]
    ### Hierarchical mergers ###
    mng = m1[a1+a2>0]
    qng = q[a1+a2>0]
    Zng = Z[a1+a2>0]
    ang = a1[a1+a2>0]

    ## Solar metallicities ##
    # Spins #
    ax = axes[0,1]

    tokeep = np.abs(np.log10(Zng/Zsun)) < 0.25
    H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=ang[tokeep], bins=[logm1bins,abins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    H[H==0] = 1e-32
    cmap = plt.get_cmap('plasma')
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),10),
                     cmap=cmap,
                     norm=norm,
                    )

    ## Add GW241011 and GW241110 posteriors ##
    """
    hdf_s241011 = h5py.File(os.path.join(GW24_dir,'standard_pe/GW241011_posterior_samples_combined.h5'))
    hdf_s241110 = h5py.File(os.path.join(GW24_dir,'standard_pe/GW241110_posterior_samples_combined.h5'))
    data_s241011 = hdf_s241011['C00:Mixed/posterior_samples']
    data_s241110 = hdf_s241110['C00:Mixed/posterior_samples']

    log_m1_241011 = np.log10(data_s241011['mass_1_source'][()])
    s1_241011 = data_s241011['a_1'][()]
    log_m1_bins_241011, s1_bins_241011, heights_241011, c90_241011 \
        = eval_kde_m1_s1(log_m1_241011, s1_241011, nbins=(100, 99), probs=(0.90,))

    tosave = np.zeros((s1_bins_241011.size+1,log_m1_bins_241011.size+1))
    tosave[1:,0] = s1_bins_241011
    tosave[0,1:] = log_m1_bins_241011
    tosave[1:,1:] = heights_241011.reshape(s1_bins_241011.size, log_m1_bins_241011.size)
    tosave[0,0] = c90_241011[0]
    np.savetxt(os.path.join(w_dir,'gw241011_am.txt'), tosave)

    log_m1_241110 = np.log10(data_s241110['mass_1_source'][()])
    s1_241110 = data_s241110['a_1'][()]
    log_m1_bins_241110, s1_bins_241110, heights_241110, c90_241110 \
    = eval_kde_m1_s1(log_m1_241110, s1_241110, nbins=(100, 99), probs=(0.90,))

    tosave = np.zeros((s1_bins_241110.size+1,log_m1_bins_241110.size+1))
    tosave[1:,0] = s1_bins_241110
    tosave[0,1:] = log_m1_bins_241110
    tosave[1:,1:] = heights_241110.reshape(s1_bins_241110.size, log_m1_bins_241110.size)
    tosave[0,0] = c90_241110[0]
    np.savetxt(os.path.join(w_dir,'gw241110_am.txt'), tosave)
    """
    gw241011_am = np.loadtxt(os.path.join(w_dir,'gw241011_am.txt'))
    s1_bins_241011 = gw241011_am[1:,0]
    log_m1_bins_241011 = gw241011_am[0,1:]
    heights_241011 = gw241011_am[1:,1:]
    c90_241011 = [gw241011_am[0,0]]

    gw241110_am = np.loadtxt(os.path.join(w_dir,'gw241110_am.txt'))
    s1_bins_241110 = gw241110_am[1:,0]
    log_m1_bins_241110 = gw241110_am[0,1:]
    heights_241110 = gw241110_am[1:,1:]
    c90_241110 = [gw241110_am[0,0]]

    ax.contourf(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
    ax.contour(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10)
    patch1 = Patch(color=gw241110_color, alpha=0.5)

    ax.contourf(log_m1_bins_241011, s1_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
    ax.contour(log_m1_bins_241011, s1_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11)
    patch2 = Patch(color=gw241011_color, alpha=0.5)

    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([])
    #ax.set_ylabel(r'$\chi_1$', fontsize=fs)
    ax.set_ylim(ymin=0., ymax=1.0)
    ax.set_yticks(np.linspace(0.,1.0,6))
    ax.set_yticklabels([])
    #ax.set_box_aspect(1)
    ax.legend(handles=(patch1,patch2), labels=('GW241110','GW241011'),
              loc='lower right', fontsize=fs,
              )
    ax.grid(color='gray', linestyle='--')
    ax.set_title(r'$Z\sim\ \mathrm{Z}_\odot$', fontsize=fs)

    # Masses #
    ax = axes[1,1]

    tokeep = np.abs(np.log10(Z1g/Zsun)) < 0.25
    print('Number of 1g+1g mergers in solar metallicity clusters:', np.sum(tokeep))
    H, xedges, yedges = np.histogram2d(x=np.log10(m1g[tokeep]), y=q1g[tokeep], bins=[logm1bins,qbins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    H /= H.sum()
    t = np.linspace(0, H.max(), n)
    integral = ((H >= t[:, None, None]) * H).sum(axis=(1,2))
    contours = interp1d(integral, t)([0.95])
    ax.contour(X, Y, H.T, contours,
               colors=CB_color_cycle[3],
               linestyles='--',
               )

    tokeep = np.abs(np.log10(Zng/Zsun)) < 0.25
    print('Number of hierarchical mergers in solar metallicity clusters:', np.sum(tokeep), '\n')
    H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=qng[tokeep], bins=[logm1bins,qbins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    H[H==0] = 1e-32
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),10),
                     cmap=cmap,
                     norm=norm,
                    )

    ## Add GW241011 and GW241110 posteriors ##
    """
    log_m1_241011 = np.log10(data_s241011['mass_1_source'][()])
    q_241011 = data_s241011['mass_ratio'][()]
    log_m1_bins_241011, q_bins_241011, heights_241011, c90_241011 \
        = eval_kde_m1_q(log_m1_241011, q_241011, nbins=(100, 99), probs=(0.90,))

    tosave = np.zeros((q_bins_241011.size+1,log_m1_bins_241011.size+1))
    tosave[1:,0] = q_bins_241011
    tosave[0,1:] = log_m1_bins_241011
    tosave[1:,1:] = heights_241011.reshape(q_bins_241011.size, log_m1_bins_241011.size)
    tosave[0,0] = c90_241011[0]
    np.savetxt(os.path.join(w_dir,'gw241011_qm.txt'), tosave)

    log_m1_241110 = np.log10(data_s241110['mass_1_source'][()])
    q_241110 = data_s241110['mass_ratio'][()]
    log_m1_bins_241110, q_bins_241110, heights_241110, c90_241110 \
    = eval_kde_m1_q(log_m1_241110, q_241110, nbins=(100, 99), probs=(0.90,))

    tosave = np.zeros((q_bins_241110.size+1,log_m1_bins_241110.size+1))
    tosave[1:,0] = q_bins_241110
    tosave[0,1:] = log_m1_bins_241110
    tosave[1:,1:] = heights_241110.reshape(q_bins_241110.size, log_m1_bins_241110.size)
    tosave[0,0] = c90_241110[0]
    np.savetxt(os.path.join(w_dir,'gw241110_qm.txt'), tosave)
    """
    gw241011_qm = np.loadtxt(os.path.join(w_dir,'gw241011_qm.txt'))
    q_bins_241011 = gw241011_qm[1:,0]
    log_m1_bins_241011 = gw241011_qm[0,1:]
    heights_241011 = gw241011_qm[1:,1:]
    c90_241011 = [gw241011_qm[0,0]]

    gw241110_qm = np.loadtxt(os.path.join(w_dir,'gw241110_qm.txt'))
    q_bins_241110 = gw241110_qm[1:,0]
    log_m1_bins_241110 = gw241110_qm[0,1:]
    heights_241110 = gw241110_qm[1:,1:]
    c90_241110 = [gw241110_qm[0,0]]

    ax.contourf(log_m1_bins_241110, q_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
    ax.contour(log_m1_bins_241110, q_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10)

    ax.contourf(log_m1_bins_241011, q_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
    ax.contour(log_m1_bins_241011, q_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11)

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([10, 30, 100])
    #ax.set_ylabel(r'$q$', fontsize=fs)
    ax.set_ylim(ymin=0.1, ymax=1.0)
    ax.set_yticks(np.linspace(0.2,1.0,5))
    ax.set_yticklabels([])
    #ax.set_box_aspect(1)
    ax.grid(color='gray', linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.045)
    cbar.set_label('Hierarchical mergers in\n my CMC clusters', fontsize=fs)
    cbar.ax.tick_params(labelsize=0.8*fs)



    ## Sub-solar metallicities ##
    # Spins #
    ax = axes[0,0]

    tokeep = np.abs(np.log10(Zng/(0.1*Zsun))) < 0.25
    H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=ang[tokeep], bins=[logm1bins,abins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    H[H==0] = 1e-32
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),10),
                     cmap=cmap,
                     norm=norm,
                    )

    ## Add GW241011 and GW241110 posteriors ##
    gw241011_am = np.loadtxt(os.path.join(w_dir,'gw241011_am.txt'))
    s1_bins_241011 = gw241011_am[1:,0]
    log_m1_bins_241011 = gw241011_am[0,1:]
    heights_241011 = gw241011_am[1:,1:]
    c90_241011 = [gw241011_am[0,0]]

    gw241110_am = np.loadtxt(os.path.join(w_dir,'gw241110_am.txt'))
    s1_bins_241110 = gw241110_am[1:,0]
    log_m1_bins_241110 = gw241110_am[0,1:]
    heights_241110 = gw241110_am[1:,1:]
    c90_241110 = [gw241110_am[0,0]]

    ax.contourf(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
    ax.contour(log_m1_bins_241110, s1_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10)

    ax.contourf(log_m1_bins_241011, s1_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
    ax.contour(log_m1_bins_241011, s1_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11)

    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([])
    ax.set_ylabel(r'$\chi_1$', fontsize=fs)
    ax.set_ylim(ymin=0., ymax=1.0)
    ax.set_yticks(np.linspace(0.,1.0,6))
    #ax.set_box_aspect(1)
    ax.grid(color='gray', linestyle='--')
    ax.set_title(r'$Z\sim 0.1\ \mathrm{Z}_\odot$', fontsize=fs)

    # Masses #
    ax = axes[1,0]

    tokeep = np.abs(np.log10(Z1g/(0.1*Zsun))) < 0.25
    print('Number of 1g+1g mergers in sub-solar metallicity clusters:', np.sum(tokeep))
    H, xedges, yedges = np.histogram2d(x=np.log10(m1g[tokeep]), y=q1g[tokeep], bins=[logm1bins,qbins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    H /= H.sum()
    t = np.linspace(0, H.max(), n)
    integral = ((H >= t[:, None, None]) * H).sum(axis=(1,2))
    contours = interp1d(integral, t)([0.95])
    ax.contour(X, Y, H.T, contours,
               colors=CB_color_cycle[3],
               linestyles='--',
               )
    ax.plot([], [],
            c=CB_color_cycle[3],
            ls='--',
            label='1g+1g',
            )

    tokeep = np.abs(np.log10(Zng/(0.1*Zsun))) < 0.25
    print('Number of hierarchical mergers in sub-solar metallicity clusters:', np.sum(tokeep))
    H, xedges, yedges = np.histogram2d(x=np.log10(mng[tokeep]), y=qng[tokeep], bins=[logm1bins,qbins])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    Hmin=H[H>0].min()
    Hmax=H[H>0].max()
    vmin = int(Hmin /10**int(np.log10(Hmin)-1))*10**(int(np.log10(Hmin))-1)
    vmax = int(Hmax /10**int(np.log10(Hmax))+1)*10**int(np.log10(Hmax))
    H[H==0] = 1e-32
    im = ax.contourf(X, Y, H.T,
                     levels=np.logspace(np.log10(vmin),np.log10(vmax),10),
                     cmap=cmap,
                     norm=norm,
                    )

    ## Add GW241011 and GW241110 posteriors ##
    gw241011_qm = np.loadtxt(os.path.join(w_dir,'gw241011_qm.txt'))
    q_bins_241011 = gw241011_qm[1:,0]
    log_m1_bins_241011 = gw241011_qm[0,1:]
    heights_241011 = gw241011_qm[1:,1:]
    c90_241011 = [gw241011_qm[0,0]]

    gw241110_qm = np.loadtxt(os.path.join(w_dir,'gw241110_qm.txt'))
    q_bins_241110 = gw241110_qm[1:,0]
    log_m1_bins_241110 = gw241110_qm[0,1:]
    heights_241110 = gw241110_qm[1:,1:]
    c90_241110 = [gw241110_qm[0,0]]

    ax.contourf(log_m1_bins_241110, q_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10, alpha=0.5)
    ax.contour(log_m1_bins_241110, q_bins_241110, heights_241110,
               levels=[*c90_241110, np.inf], colors=gw241110_color, zorder=10)

    ax.contourf(log_m1_bins_241011, q_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11, alpha=0.5)
    ax.contour(log_m1_bins_241011, q_bins_241011, heights_241011,
               levels=[*c90_241011, np.inf], colors=gw241011_color, zorder=11)

    ax.set_xlabel(r'$m_1\ [\mathrm{M}_\odot]$', fontsize=fs)
    ax.set_xlim(np.log10(5), np.log10(120))
    ax.set_xticks(np.log10([10, 30, 100]))
    ax.set_xticklabels([10, 30, 100])
    ax.set_ylabel(r'$q$', fontsize=fs)
    ax.set_ylim(ymin=0.1, ymax=1.0)
    ax.set_yticks(np.linspace(0.2,1.0,5))
    #ax.set_box_aspect(1)
    ax.legend(loc='lower left', fontsize=fs)
    ax.grid(color='gray', linestyle='--')


    # Save
    plot_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical/plots'
    plt.savefig(os.path.join(plot_dir,'Fig6_allsims.png'), bbox_inches='tight')



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    FIRE_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2'
    grid_dir = os.path.join(FIRE_dir,'grid')
    sim_names = ['m12i','m11d','m11h','m11e','m11q','m11i']

    w_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/hierarchical'
    GW24_dir = '/home/tristan-bruel/Documents/Science/o4a-astro/GW241011_GW241110_data_release/data'
    gw241011_color = '#1f78b4'
    gw241110_color = '#33a02c'

    plot_mBH_spin()
    plt.show()
