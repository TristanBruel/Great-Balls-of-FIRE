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


def load_bbhs(withKremer=True, verbose=False):
    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'
    bbhfile = os.path.join(grid_dir, 'allBBHs' +withKremer*'_withKremer'+'.dat')
    bh_cl_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=np.str_)
    unique_cl_id, nbbhs = np.unique(bh_cl_id, return_counts=True)
    mbh1, mbh2, abh1, abh2 = np.loadtxt(bbhfile, usecols=[4,5,8,9]).T
    abh1[mbh1<mbh2], abh2[mbh1<mbh2] = abh2[mbh1<mbh2], abh1[mbh1<mbh2]
    mbh1, mbh2 = np.maximum(mbh1,mbh2), np.minimum(mbh1,mbh2)
    qbbh = mbh2 /mbh1 
    hierar = abh1+abh2 > 0
    if verbose:
        print('Total number of CMC clusters: %i' %(len(nbbhs)))
        print('Total number of BBH mergers in CMC clusters: {}'.format(len(mbh1)))
        print('Number of hierarchical BBH mergers: {}\n'.format(np.sum(hierar)))
    bbhs = {}
    bbhs['1g1g'] = {'nbbhs': np.sum(~hierar), 
                    'm1': mbh1[~hierar], 
                    'q': qbbh[~hierar], 
                    'a1': abh1[~hierar],
                    }
    bbhs['hierar'] = {'nbbhs': np.sum(hierar), 
                    'm1': mbh1[hierar], 
                    'q': qbbh[hierar], 
                    'a1': abh1[hierar],
                    }
    return bbhs



def compute_logB(bbhs, PE, pri_at_pos_kde, event=None, astro=False, verbose=False):
    logB = np.zeros(len(bbhs))
    for n,c in enumerate(bbhs.keys()):
        m1 = bbhs[c]['m1']
        q = bbhs[c]['q']
        #a1 = bbhs[c]['a1']
        kde_n = KDE([m1, q])
        #kde_n = KDE([m1, q, a1])
        model_n = lambda x: kde_n.pdf(x)
        x = [PE['mass_1_source'], PE['mass_ratio']]
        #x = [PE['mass_1_source'], PE['mass_ratio'], PE['a_1']]
        log_bfAB_nonorm = popodds.log_odds(
            model_n, pri_at_pos_kde, x,
            log=False,
        )
        logB[n] = log_bfAB_nonorm
    print('Event:', event)
    if astro:
        print('Astrophysically motivated priors')
    else:
        print('Standard LVK priors')
    print('log_BF_{hierar/1g+1g} = %.2f' %(logB[1]-logB[0]))
    #outfile = 'logBF_%s_PE' %event + astro*'_astro' + '.dat'
    #header = 'cluster_id\t logBf/PE'
    #data=np.vstack([np.array(list(clusters.keys())),logB], dtype=object).T
    #fmt='%s','%f'
    #np.savetxt(outfile, data, header=header, delimiter='\t', fmt=fmt)
    return logB



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", dest='event', help="GW event to compute Bayes factor for", type=str, default='GW241011')
    parser.add_argument("--Kremer", action='store_true', help="Include CMC clusters from Kremer et al. 2018")
    parser.add_argument("--astro", dest='astro_prior', action='store_true', help="Uses PE samples with astro priors")
    parser.add_argument("-v", "--verbose", action='store_true', help="Add print statements")
    args = parser.parse_args()

    PE, prior_samples = load_PE_samples(event=args.event, astro=args.astro_prior)
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


    bbhs = load_bbhs(withKremer=args.Kremer, verbose=args.verbose)
    logB = compute_logB(bbhs, PE, pri_at_pos_kde, 
                        event=args.event, astro=args.astro_prior,
                        verbose=args.verbose)
