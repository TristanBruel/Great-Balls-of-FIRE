import numpy as np
import h5py 
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde


import os
import argparse



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



def load_PE_samples(event='GW241011', astro=False):
    myPE_dir = '/home/tristan-bruel/Documents/Science/LVK/PE'
    if astro:
        #file_PE = os.path.join(myPE_dir, event, 'astro_pop_prior/web/samples/posterior_samples.h5')
        #with h5py.File(file_PE, 'r') as h:
        #    pos = h['C00:IMRPhenomXPHM']['posterior_samples']
        #    posterior_samples = {
        #                         'chirp_mass': pos['chirp_mass'][:],
        #                         'mass_1_source': pos['mass_1_source'][:],
        #                         'mass_ratio': pos['mass_ratio'][:],
        #                         'a_1': pos['a_1'][:],
        #                         }
        file_PE = os.path.join(myPE_dir, event, 'astro_pop_prior', event+'_astro_merged_result.hdf5')
        with h5py.File(file_PE, 'r') as h:
            pos = h['posterior']
            posterior_samples = {
                                 'chirp_mass': pos['chirp_mass'][:],
                                 'mass_1_source': pos['mass_1_source'][:],
                                 'mass_ratio': pos['mass_ratio'][:],
                                 'a_1': pos['a_1'][:],
                                 'redshift': pos['redshift'][:],
                                 }
    else:
        #file_PE = os.path.join(myPE_dir, event, 'standard/web/samples/posterior_samples.h5')
        #with h5py.File(file_PE, 'r') as h:
        #    pos = h['C00:IMRPhenomXPHM']['posterior_samples']
        #    posterior_samples = {
        #                         'chirp_mass': pos['chirp_mass'][:],
        #                         'mass_1_source': pos['mass_1_source'][:],
        #                         'mass_ratio': pos['mass_ratio'][:],
        #                         'a_1': pos['a_1'][:],
        #                         }
        file_PE = os.path.join(myPE_dir, event, 'standard', event+'_merged_result.hdf5')
        with h5py.File(file_PE, 'r') as h:
            pos = h['posterior']
            posterior_samples = {
                                 'chirp_mass': pos['chirp_mass'][:],
                                 'mass_1_source': pos['mass_1_source'][:],
                                 'mass_ratio': pos['mass_ratio'][:],
                                 'a_1': pos['a_1'][:],
                                 'redshift': pos['redshift'][:],
                                 }
    return posterior_samples 



def save_contours(PE, event, astro=False): 
    log_m1 = np.log10(PE['mass_1_source'])
    q = PE['mass_ratio']
    s1 = PE['a_1']

    log_m1_bins, s1_bins, heights, c90 \
        = eval_kde_m1_s1(log_m1, s1, nbins=(50, 49), probs=(0.90,))
    tosave = np.zeros((s1_bins.size+1,log_m1_bins.size+1))
    tosave[1:,0] = s1_bins
    tosave[0,1:] = log_m1_bins
    tosave[1:,1:] = heights.reshape(s1_bins.size, log_m1_bins.size)
    tosave[0,0] = c90[0]
    w_dir = './contours'
    np.savetxt(os.path.join(w_dir,event+'_am'+astro*'_astro'+'.txt'), tosave)

    log_m1_bins, q_bins, heights, c90\
        = eval_kde_m1_q(log_m1, q, nbins=(50, 49), probs=(0.90,))
    tosave = np.zeros((q_bins.size+1,log_m1_bins.size+1))
    tosave[1:,0] = q_bins
    tosave[0,1:] = log_m1_bins
    tosave[1:,1:] = heights.reshape(q_bins.size, log_m1_bins.size)
    tosave[0,0] = c90[0]
    np.savetxt(os.path.join(w_dir,event+'_qm'+astro*'_astro'+'.txt'), tosave)



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", dest='event', help="GW event to compute Bayes factor for", type=str, default='GW241011')
    parser.add_argument("--astro", dest='astro_prior', action='store_true', help="Uses PE samples with astro priors")
    args = parser.parse_args()

    PE = load_PE_samples(event=args.event, astro=args.astro_prior)
    save_contours(PE, event=args.event, astro=args.astro_prior)
