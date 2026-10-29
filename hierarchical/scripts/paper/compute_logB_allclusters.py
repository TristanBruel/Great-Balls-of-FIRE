import numpy as np
import h5py 
from scipy.interpolate import interp1d
from scipy.integrate import quad
from kaydee import KDE
import popodds
import bilby

import os
import argparse
from tqdm import tqdm


def load_PE_samples(event='GW241011', astro=False):
    myPE_dir = '/home/tristan-bruel/Documents/Science/LVK/PE'
    if astro:
        """
        file_PE = os.path.join(myPE_dir, event, 'astro_pop_prior/web/samples/posterior_samples.h5')
        with h5py.File(file_PE, 'r') as h:
            pos = h['C00:IMRPhenomXPHM']['posterior_samples']
            posterior_samples = {
                                 'chirp_mass': pos['chirp_mass'][:],
                                 'mass_1_source': pos['mass_1_source'][:],
                                 'mass_ratio': pos['mass_ratio'][:],
                                 'a_1': pos['a_1'][:],
                                 'redshift': pos['redshift'][:],
                                 }
        """
        file_PE = os.path.join(myPE_dir, event, 'astro_pop_prior/%s_astro_merged_result.hdf5' %event)
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
        """
        file_PE = os.path.join(myPE_dir, event, 'standard/web/samples/posterior_samples.h5')
        with h5py.File(file_PE, 'r') as h:
            pos = h['C00:IMRPhenomXPHM']['posterior_samples']
            posterior_samples = {
                                 'chirp_mass': pos['chirp_mass'][:],
                                 'mass_1_source': pos['mass_1_source'][:],
                                 'mass_ratio': pos['mass_ratio'][:],
                                 'a_1': pos['a_1'][:],
                                 'redshift': pos['redshift'][:],
                                 }
        """
        file_PE = os.path.join(myPE_dir, event, 'standard/%s_merged_result.hdf5' %event)
        with h5py.File(file_PE, 'r') as h:
            # Posterior samples #
            pos = h['posterior']
            posterior_samples = {
                                 'chirp_mass': pos['chirp_mass'][:],
                                 'mass_1_source': pos['mass_1_source'][:],
                                 'mass_ratio': pos['mass_ratio'][:],
                                 'a_1': pos['a_1'][:],
                                 'redshift': pos['redshift'][:],
                                 }
    return posterior_samples 



def pri_at_pos(event, astro=False, nospin=False):
    PE = load_PE_samples(event=event, astro=astro)
    PE_dir = os.path.join('/home/tristan-bruel/Documents/Science/LVK/PE', event)
    if astro:
        prior_file = os.path.join(PE_dir, 'astro_pop_prior/'+event+'_prior.json')
        prior = bilby.gw.prior.BBHPriorDict.from_json(prior_file)
        if nospin:
            prior_evals = np.exp([
                prior.ln_prob(
                    {
                        'mass_1_source': PE['mass_1_source'][i],
                        'mass_ratio': PE['mass_ratio'][i],
                        'redshift': PE['redshift'][i],
                     },
                    normalized=False,
                )
                - prior['redshift'].ln_prob(PE['redshift'][i])
                for i in range(len(PE['mass_1_source']))
            ])
        else:
            prior_evals = np.exp([
                prior.ln_prob(
                    {
                        'mass_1_source': PE['mass_1_source'][i],
                        'mass_ratio': PE['mass_ratio'][i],
                        'a_1': PE['a_1'][i],
                        'redshift': PE['redshift'][i],
                     },
                    normalized=False,
                )
                - prior['redshift'].ln_prob(PE['redshift'][i])
                for i in range(len(PE['mass_1_source']))
            ])
        return prior_evals
    else:
        prior_file = os.path.join(PE_dir, 'standard/'+event+'_prior.json')
        prior = bilby.gw.prior.BBHPriorDict.from_json(prior_file)
        if nospin:
            prior_evals = np.exp([
                prior.ln_prob(
                    {
                        'chirp_mass': PE['chirp_mass'][i],
                        'mass_ratio': PE['mass_ratio'][i],
                     },
                    normalized=False,
                )
                for i in range(len(PE['chirp_mass']))
            ])
        else:
            prior_evals = np.exp([
                prior.ln_prob(
                    {
                        'chirp_mass': PE['chirp_mass'][i],
                        'mass_ratio': PE['mass_ratio'][i],
                        'a_1': PE['a_1'][i],
                     },
                    normalized=False,
                )
                for i in range(len(PE['chirp_mass']))
            ])
        return prior_evals




def load_clusters(nmin=2, withKremer=True, verbose=False):
    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'
    bbhfile = os.path.join(grid_dir, 'allBBHs' +withKremer*'_withKremer'+'.dat')
    cmc_cluster_ids = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=np.str_)
    unique_cl_id, nbbhs = np.unique(cmc_cluster_ids, return_counts=True)
    mbh1, mbh2, abh1, abh2 = np.loadtxt(bbhfile, usecols=[4,5,8,9]).T
    abh1[mbh1<mbh2], abh2[mbh1<mbh2] = abh2[mbh1<mbh2], abh1[mbh1<mbh2]
    mbh1, mbh2 = np.maximum(mbh1,mbh2), np.minimum(mbh1,mbh2)
    qbbh = mbh2 /mbh1 
    if verbose:
        print('Total number of BBH mergers in CMC clusters: {}'.format(len(mbh1)))
        print('Number of hierarchical BBH mergers: {}\n'.format(np.sum(abh1+abh2>0)))
    tokeep = nbbhs>nmin
    if verbose:
        print('Number of CMC clusters with at least %i BBH mergers: %i' %(nmin,np.sum(tokeep)))
    clusters = {}
    for c in unique_cl_id[tokeep]:
        clusters[str(c)] = {'nbbhs': nbbhs[unique_cl_id==c][0], 
                            'm1': mbh1[cmc_cluster_ids==c],
                            'q': qbbh[cmc_cluster_ids==c],
                            'a1': abh1[cmc_cluster_ids==c],
                            }
    return clusters



def compute_logB(clusters, PE, pri_at_pos, event=None, 
                 withKremer=False, astro=False, nospin=False, 
                 verbose=False):
    logB = np.zeros(len(clusters))
    for n,c in enumerate(clusters.keys()):
        m1 = clusters[c]['m1']
        q = clusters[c]['q']
        Mc = m1 *q**(3/5) /(1+q)**(1/5)
        a1 = clusters[c]['a1']
        if nospin:
            if astro:
                x = [PE['mass_1_source'], PE['mass_ratio']]
                kde_n = KDE([m1, q])
            else:
                x = [PE['chirp_mass'], PE['mass_ratio']]
                kde_n = KDE([Mc, q])
            model_n = lambda x: kde_n.pdf(x)
        else:
            if np.sum(a1>0)==0:
                logB[n] = -np.inf
                continue
            else:
                if astro:
                    x = [PE['mass_1_source'], PE['mass_ratio'], PE['a_1']]
                    kde_n = KDE([m1, q, a1])
                else:
                    x = [PE['chirp_mass'], PE['mass_ratio'], PE['a_1']]
                    kde_n = KDE([Mc, q, a1])
            model_n = lambda x: kde_n.pdf(x)

        log_bfAB_nonorm = popodds.log_odds(
            model_n, pri_at_pos, x,
            log=False,
        )
        if verbose:
            print('log BF_{%s/PE} = %f' %(c,log_bfAB_nonorm))
        logB[n] = log_bfAB_nonorm

    filename = 'logBF_%s_PE' %event + withKremer*'_withKremer' + astro*'_astro' + nospin*'_nospin' + '.dat'
    outfile = os.path.join('log_Bayes_factors',filename)
    header = 'cluster_id\t logBf/PE'
    data=np.vstack([np.array(list(clusters.keys())),logB], dtype=object).T
    fmt='%s','%f'
    np.savetxt(outfile, data, header=header, delimiter='\t', fmt=fmt)
    return logB



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", dest='event', help="GW event to compute Bayes factor for", type=str, default='GW241011')
    parser.add_argument("--nmin", dest='nmin', help="Minimum number of BBH mergers in given cluster to compute logB", type=int, default=3)
    parser.add_argument("--Kremer", action='store_true', help="Include CMC clusters from Kremer et al. 2018")
    parser.add_argument("--astro", dest='astro_prior', action='store_true', help="Uses PE samples with astro priors")
    parser.add_argument("--nospin", dest='nospin', action='store_true', help="Do not use spins to compute Bayes factors")
    parser.add_argument("-v", "--verbose", action='store_true', help="Add print statements")
    args = parser.parse_args()

    PE = load_PE_samples(event=args.event, astro=args.astro_prior)
    pri_evals = pri_at_pos(event=args.event, astro=args.astro_prior, nospin=args.nospin)
    pri_at_pos_evals = lambda x: pri_evals

    clusters = load_clusters(nmin=args.nmin, withKremer=args.Kremer, verbose=args.verbose)
    logB = compute_logB(clusters, PE, pri_at_pos_evals, event=args.event, 
                        withKremer=args.Kremer, astro=args.astro_prior, nospin=args.nospin,
                        verbose=args.verbose)
