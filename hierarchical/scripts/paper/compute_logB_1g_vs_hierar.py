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
        file_PE = os.path.join(myPE_dir, event, 'astro_pop_prior/%s_astro_merged_result.hdf5' %event)
        with h5py.File(file_PE, 'r') as h:
            pos = h['posterior']
            posterior_samples = {
                                 'chirp_mass': pos['chirp_mass'][:],
                                 'mass_1_source': pos['mass_1_source'][:],
                                 'mass_ratio': pos['mass_ratio'][:],
                                 'a_1': pos['a_1'][:],
                                 'a_2': pos['a_2'][:],
                                 'redshift': pos['redshift'][:],
                                 }
    else:
        file_PE = os.path.join(myPE_dir, event, 'standard/%s_merged_result.hdf5' %event)
        with h5py.File(file_PE, 'r') as h:
            # Posterior samples #
            pos = h['posterior']
            posterior_samples = {
                                 'chirp_mass': pos['chirp_mass'][:],
                                 'mass_1_source': pos['mass_1_source'][:],
                                 'mass_ratio': pos['mass_ratio'][:],
                                 'a_1': pos['a_1'][:],
                                 'a_2': pos['a_2'][:],
                                 'redshift': pos['redshift'][:],
                                 }
    return posterior_samples 



def pri_at_pos(event, astro=False):
    PE = load_PE_samples(event=event, astro=astro)
    PE_dir = os.path.join('/home/tristan-bruel/Documents/Science/LVK/PE', event)
    if astro:
        prior_file = os.path.join(PE_dir, 'astro_pop_prior/'+event+'_prior.json')
        prior = bilby.gw.prior.BBHPriorDict.from_json(prior_file)
        prior_evals = np.exp([
            prior.ln_prob(
                {
                    'mass_1_source': PE['mass_1_source'][i],
                    'mass_ratio': PE['mass_ratio'][i],
                    'a_1': PE['a_1'][i],
                    #'a_2': PE['a_2'][i],
                    'redshift': PE['redshift'][i],
                 },
                normalized=False,
            )
            - prior['redshift'].ln_prob(PE['redshift'][i])
            for i in range(len(PE['mass_1_source']))
        ])
        jac = (1+PE['redshift']) *PE['mass_ratio']**(3/5) /(1+PE['mass_ratio'])**(1/5) # dMc / dm1s
        prior_evals /= jac
        return prior_evals
    else:
        prior_file = os.path.join(PE_dir, 'standard/'+event+'_prior.json')
        prior = bilby.gw.prior.BBHPriorDict.from_json(prior_file)
        prior_evals = np.exp([
            prior.ln_prob(
                {
                    'chirp_mass': PE['chirp_mass'][i],
                    'mass_ratio': PE['mass_ratio'][i],
                    'a_1': PE['a_1'][i],
                    #'a_2': PE['a_2'][i],
                 },
                normalized=False,
            )
            for i in range(len(PE['chirp_mass']))
        ])
        return prior_evals



def load_bbhs(withKremer=True):
    grid_dir = '/home/tristan-bruel/Documents/Science/Great-Balls-of-FIRE/FIRE-2/grid'
    bbhfile = os.path.join(grid_dir, 'allBBHs' +withKremer*'_withKremer'+'.dat')
    bh_cl_id = np.loadtxt(bbhfile, usecols=0, skiprows=1, dtype=np.str_)
    unique_cl_id, nbbhs = np.unique(bh_cl_id, return_counts=True)
    mbh1, mbh2, abh1, abh2 = np.loadtxt(bbhfile, usecols=[4,5,8,9]).T
    abh1[mbh1<mbh2], abh2[mbh1<mbh2] = abh2[mbh1<mbh2], abh1[mbh1<mbh2]
    mbh1, mbh2 = np.maximum(mbh1,mbh2), np.minimum(mbh1,mbh2)
    qbbh = mbh2 /mbh1 
    Mcbbh = (mbh1*mbh2)**(3/5) /(mbh1+mbh2)**(1/5)
    hierar = abh1+abh2 > 0
    bbhs = {}
    bbhs['1g1g'] = {'nbbhs': np.sum(~hierar), 
                    'Mc': Mcbbh[~hierar], 
                    'q': qbbh[~hierar], 
                    'a1': abh1[~hierar],
                    'a2': abh2[~hierar],
                    }
    bbhs['hierar'] = {'nbbhs': np.sum(hierar), 
                    'Mc': Mcbbh[hierar], 
                    'q': qbbh[hierar], 
                    'a1': abh1[hierar],
                    'a2': abh2[hierar],
                    }
    return bbhs



def compute_logB(bbhs, PE, pri_at_pos, event=None, 
                 astro=False, 
                 ):
    logB = np.zeros(len(bbhs))
    for n,k in enumerate(bbhs.keys()):
        Mc = bbhs[k]['Mc']
        q = bbhs[k]['q']
        a1 = bbhs[k]['a1']
        a2 = bbhs[k]['a2']
        x = [PE['chirp_mass'], PE['mass_ratio'], PE['a_1']]
        kde_n = KDE([Mc, q, a1])
        #x = [PE['chirp_mass'], PE['mass_ratio'], PE['a_1'], PE['a_2']]
        #kde_n = KDE([Mc, q, a1, a2])
        model_n = lambda x: kde_n.pdf(x)

        log_bfAB_nonorm = popodds.log_odds(
            model_n, pri_at_pos, x,
            log=False,
        )
        logB[n] = log_bfAB_nonorm
    if astro:
        print('\tAstrophysically motivated priors')
    else:
        print('\tStandard LVK priors')
    print('\tlog_BF_{hierar/1g+1g} = %.2f\n' %(logB[1]-logB[0]))
    return logB



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", help="GW event to consider", nargs='+', default=['GW241011','GW241110'])
    parser.add_argument("--Kremer", action='store_true', help="Include CMC clusters from Kremer et al. 2018")
    args = parser.parse_args()

    bbhs = load_bbhs(withKremer=args.Kremer)

    for event in args.events:
        print('Computing 1g+1g vs hierarhical Bayes factor for event:', event)
        for astro in [False,True]:
            PE = load_PE_samples(event=event, astro=astro)
            pri_evals = pri_at_pos(event=event, astro=astro)
            pri_at_pos_evals = lambda x: pri_evals

            logB = compute_logB(bbhs, PE, pri_at_pos_evals, 
                                event=event, astro=astro,
                                )
