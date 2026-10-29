import numpy as np
import h5py 
from scipy.interpolate import interp1d
from scipy.integrate import quad
from kaydee import KDE
import popodds

import os
import argparse
from tqdm import tqdm


def load_PE_samples(event='GW241011', astro=False):
    myPE_dir = '/home/tristan-bruel/Documents/Science/LVK/PE'
    if astro:
        file_PE = os.path.join(myPE_dir, event, 'astro_pop_prior/web/samples/posterior_samples.h5')
        with h5py.File(file_PE, 'r') as h:
            # Posterior samples #
            pos = h['C00:IMRPhenomXPHM']['posterior_samples']
            posterior_samples = {'mass_1_source': pos['mass_1_source'][:],
                                 'mass_ratio': pos['mass_ratio'][:],
                                 'a_1': pos['a_1'][:],
                                 }
            # Prior samples #
            pri = h['C00:IMRPhenomXPHM']['priors']['samples']
            prior_samples = {'mass_1_source': pri['mass_1_source'][:],
                             'mass_2_source': pri['mass_1_source'][:] *pri['mass_ratio'][:],
                             'a_1': pri['a_1'][:],
                             }
    else:
        file_PE = os.path.join(myPE_dir, event, 'standard/web/samples/posterior_samples.h5')
        with h5py.File(file_PE, 'r') as h:
            # Posterior samples #
            pos = h['C00:IMRPhenomXPHM']['posterior_samples']
            posterior_samples = {'mass_1_source': pos['mass_1_source'][:],
                                 'mass_ratio': pos['mass_ratio'][:],
                                 'a_1': pos['a_1'][:],
                                 }
            # Prior samples #
            pri = h['C00:IMRPhenomXPHM']['priors']['samples']
            prior_samples = {'mass_1_source': pri['mass_1_source'][:],
                             'mass_2_source': pri['mass_2_source'][:],
                             'a_1': pri['a_1'][:],
                             }
    return posterior_samples, prior_samples



def pri_at_pos_kde(x, kde_prior):
    m1z, q, a1 = x
    m2z = m1z *q
    return (
            kde_prior.pdf([m1z, m2z, a1])
            )


def load_clusters(nmin=2, withKremer=True, verbose=False):
    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'
    bbhfile = os.path.join(grid_dir, 'allBBHs' +withKremer*'_withKremer'+'.dat')
    bh_cl_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=np.str_)
    unique_cl_id, nbbhs = np.unique(bh_cl_id, return_counts=True)
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
                            'm1': mbh1[bh_cl_id==c],
                            'q': qbbh[bh_cl_id==c],
                            'a1': abh1[bh_cl_id==c],
                            }
    return clusters



def compute_logB(clusters, PE, pri_at_pos_kde, event=None, 
                 withKremer=False, astro=False, nospin=False, 
                 verbose=False):
    logB = np.zeros(len(clusters))
    for n,c in enumerate(clusters.keys()):
        m1 = clusters[c]['m1']
        q = clusters[c]['q']
        a1 = clusters[c]['a1']
        if np.sum(a1>0)==0 and ~nospin:
            logB[n] = -np.inf
        else:
            if nospin:
                kde_n = KDE([m1, q])
            else:
                kde_n = KDE([m1, q, a1])
            model_n = lambda x: kde_n.pdf(x)
            if nospin:
                x = [PE['mass_1_source'], PE['mass_ratio']]
            else:
                x = [PE['mass_1_source'], PE['mass_ratio'], PE['a_1']]
            log_bfAB_nonorm = popodds.log_odds(
                model_n, pri_at_pos_kde, x,
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

    PE, prior_samples = load_PE_samples(event=args.event, astro=args.astro_prior)
    if args.nospin:
        # Uniform volumetric merger rate in the source frame
        kde_prior = KDE([prior_samples['mass_1_source'], prior_samples['mass_2_source']])
        def pri_at_pos_kde(x, kde_prior):
            m1z, q = x
            m2z = m1z *q
            return (
                    kde_prior.pdf([m1z, m2z])
                    )
        kde_evals = pri_at_pos_kde([PE['mass_1_source'], PE['mass_ratio']], kde_prior)
        pri_at_pos_kde = lambda x: kde_evals
    else:
        # Uniform volumetric merger rate in the source frame
        kde_prior = KDE([prior_samples['mass_1_source'], prior_samples['mass_2_source'], prior_samples['a_1']])
        def pri_at_pos_kde(x, kde_prior):
            m1z, q, a1 = x
            m2z = m1z *q
            return (
                    kde_prior.pdf([m1z, m2z, a1])
                    )
        kde_evals = pri_at_pos_kde([PE['mass_1_source'], PE['mass_ratio'], PE['a_1']], kde_prior)
        pri_at_pos_kde = lambda x: kde_evals


    clusters = load_clusters(nmin=args.nmin, withKremer=args.Kremer, verbose=args.verbose)
    logB = compute_logB(clusters, PE, pri_at_pos_kde, event=args.event, 
                        withKremer=args.Kremer, astro=args.astro_prior, nospin=args.nospin,
                        verbose=args.verbose)
