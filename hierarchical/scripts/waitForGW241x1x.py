import numpy as np
import h5py 
from scipy.interpolate import interp1d
from tqdm import tqdm

import os
import sys
from glob import glob

from colossus.cosmology import cosmology
cosmology.setCosmology('planck18');
from colossus.lss import mass_function
from astropy.cosmology import Planck18 as planck
from gwbench import Network, M_of_Mc_eta, f_isco_Msolar, injections

import popsummary
dir_GWTC4 = '/home/tristan-bruel/Documents/Science/o4a-astro'

FIREbox_dir = '/home/tristan-bruel/Documents/Science/GBoF/FIREbox'
FIREbox_volume = (22.1)**3 # in unit Mpc^3
Tobs = 1 # LVK observation time in yrs


def HaloMassFunction():
    dndM = lambda M: mass_function.massFunction(M, 0.0, mdef='vir', model='tinker08', q_out='dndlnM')
    return dndM


def sample_detections(sample='SFR'):

    snapshot_file = os.path.join(FIREbox_dir, 'snapshot_times.txt')
    redshifts, times = np.loadtxt(snapshot_file, usecols=[2,3]).T

    times_to_redshifts = interp1d(times, redshifts, fill_value='extrapolate')
    redshifts_to_times = interp1d(redshifts, times, fill_value='extrapolate')
    #tmax = times.max()
    #deltat = 150 # width of the time bins (in Myr)
    #nbins = round(1e3*tmax /deltat) + 1
    #tbins = np.linspace(0, tmax, nbins)
    zbins = np.linspace(0, 3, 101)
    #zbins = times_to_redshifts(tbins)
    tbins = redshifts_to_times(zbins)

    Mstar_file = os.path.join(FIREbox_dir, 'Mstar90_v2.dat')
    haloids, Mvir, Mstar = np.loadtxt(Mstar_file, usecols=[0,1,3], skiprows=1).T
    haloids = haloids.astype(np.int64)

    logMmin=10.
    logMmax=14.
    to_keep = (Mvir>np.power(10,logMmin))&(Mvir<np.power(10,logMmax))
    print('Number of galaxies identified in FIREbox with halo mass in [10^{},10^{}]: {}'.format(logMmin,logMmax,np.sum(to_keep)))
    Mvir = Mvir[to_keep]
    Mstar = Mstar[to_keep]
    haloids = haloids[to_keep]

    nbins = round((logMmax-logMmin)*5 + 1)
    logmbins = np.linspace(logMmin, logMmax, nbins)
    mid_bins = (logmbins[:-1]+logmbins[1:]) /2
    Nhalo, _ = np.histogram(np.log10(Mvir), bins=logmbins)
    dnhalo = Nhalo /np.diff(logmbins) /FIREbox_volume
    wgal = np.zeros_like(dnhalo)
    obs_HMF = HaloMassFunction() # number density, unit is Mpc^-3
    wgal[dnhalo>0] = obs_HMF(np.power(10,mid_bins))[dnhalo>0] /dnhalo[dnhalo>0]


    ## Load GWTC-4 results ##
    file_bptp04 = os.path.join(dir_GWTC4, 'data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5')
    #file_bs04 = os.path.join(dir_GWTC4, 'data_release/BBHMassSpinRedshift_BSplineIID.h5')
    bptp_o4 = popsummary.popresult.PopulationResult(file_bptp04)
    #bs_o4 = popsummary.popresult.PopulationResult(file_bs04)

    z_list, R_z = bptp_o4.get_rates_on_grids('redshift')
    z_list = z_list[0]
    #bs_z_list, bs_R_z = bs_o4.get_rates_on_grids('rate_vs_redshift')
    #bs_z_list = bs_z_list[0]

    # Extrapolate some cosmology quantities (dz/dt and dVc/dz)
    dzdt = interp1d(z_list[:-1], np.diff(z_list)/np.diff(planck.lookback_time(z_list)), fill_value='extrapolate')
    dVcdz = interp1d(z_list, (4*np.pi)*planck.differential_comoving_volume(z_list).value*1e-9, fill_value='extrapolate')
    V15 = np.trapezoid(x=z_list, y=dVcdz(z_list))
    print('dVcdz integrated up to z=%.2f: %.2e Gpc^-3' %(z_list[-1],V15))

    dNdz = R_z *Tobs/(1+z_list) *dVcdz(z_list)
    Ntot = np.trapezoid(x=z_list, y=np.percentile(dNdz,50,axis=0))
    print('Total number of BBH mergers up to redshift 1.5 (GWTC-4.0): %i' %Ntot)
    print('\n')


    #bbh_file = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_seed9.h5'.format(sample))
    bbh_file = os.path.join(FIREbox_dir, 'BBHs/BBHs_{}_compiled_withKremer_seed9.h5'.format(sample))
    print(bbh_file)
    with h5py.File(bbh_file, 'r') as bbhs:
        m1 = bbhs['m1'][()]
        m2 = bbhs['m2'][()]
        a1 = bbhs['a1'][()]
        a2 = bbhs['a2'][()]
        tmerger = bbhs['tmerger'][()]
        Mgal = bbhs['Mgal'][()]
        Mhalo = bbhs['Mvir'][()]
    zmerger = times_to_redshifts(tmerger)
    astro = (zmerger>0) & (zmerger<=1.5)
    m1 = m1[astro]
    m2 = m2[astro]
    a1 = a1[astro]
    a2 = a2[astro]
    Mhalo = Mhalo[astro]
    tmerger = tmerger[astro]
    zmerger = zmerger[astro]
    print('Unweighted number of BBH mergers in FIREbox at z<1.5: %i' %len(m1))
    a1[m1<m2], a2[m1<m2] = a2[m1<m2], a1[m1<m2]
    m1, m2 = np.maximum(m1,m2), np.minimum(m1,m2)
    q = m2 /m1

    # Compute weight associated to each galaxy in which there's a BBH merger
    weights = np.array([wgal[k-1] for k in np.digitize(np.log10(Mhalo), logmbins)])
    weights *= 1/FIREbox_volume *Tobs/(1+zmerger) *dVcdz(zmerger) *dzdt(zmerger)
    print('Estimated number of BBH mergers up to z=1.5 over 1 year of observation time at LIGO design sensitivity')
    print('%i' %np.sum(weights))

    ## BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses
    ## i.e. compatible with GW241011 and GW241110
    print('\n')
    print('Looking for BBH mergers with stricly positive primary spin, m1 < 30 Msun, and unequal masses')
    ofinterest = (m1 < 30) & (q < 0.85) & (a1 > 0.05)
    print('Unweighted number of these BBH mergers in FIREbox at z<1.5: %i' %np.sum(ofinterest))
    print('Estimated number of these BBH mergers up to z=1.5 over 1 year of observation time at LIGO design sensitivity')
    print('%i' %np.sum(weights[ofinterest]))

    ## Find how many could be detectable ##
    m1s = m1[ofinterest]
    m2s = m2[ofinterest]
    a1s = a1[ofinterest]
    a2s = a2[ofinterest]
    zs = zmerger[ofinterest]
    isobservable = np.zeros(len(m1s), dtype=bool)

    ## Compute SNRs ##
    deriv_symbs_string  = 'Mc eta chi1z chi2z DL tc phic iota ra dec psi'
    conv_log            = ('Mc','DL')
    use_rot             = True
    only_net            = True
    # initialize the network with the desired detectors
    #network_spec = ['aLIGO_LHO', 'aLIGO_LLO']
    # single detector
    network_spec = ['aLIGO_LHO']
    nnet = Network(network_spec,logger_name='Network',logger_level='WARNING')
    # pick the desired frequency range
    f_lo = 1.
    df   = 2.**-4

    iota_vec, ra_vec, dec_vec, psi_vec = injections.angle_sampler(len(m1s), seed=42)
    for i in tqdm(range(len(m1s))):
        Mc = (m1s[i]*m2s[i])**(3/5) /(m1s[i]+m2s[i])**(1/5)
        eta = m1s[i]*m2s[i] /(m1s[i]+m2s[i])**2
        chi1 = a1s[i]
        chi2 = a2s[i]
        Dl = planck.luminosity_distance(z=zs[i]).value # in unit Mpc
        inj_params = {
            'Mc'    : Mc,
            'eta'   : eta,
            'chi1x' : chi1,
            'chi1y' : 0.0,
            'chi1z' : 0.0,
            'chi2x' : 0.0,
            'chi2y' : chi2,
            'chi2z' : 0.0,
            'DL'    : Dl,
            'tc'    : 0.0,
            'phic'  : 0.0,
            'iota'  : iota_vec[i],
            'ra'    : ra_vec[i],
            'dec'   : dec_vec[i],
            'psi'   : psi_vec[i],
        }
        #print('injections parameter: ', inj_params)
        f_hi = f_isco_Msolar(M_of_Mc_eta(inj_params['Mc'],inj_params['eta']))
        f    = np.arange(f_lo,f_hi+df,df)
        nnet.reset_det_responses()
        nnet.set_net_vars(wf_model_name='tf2', f=f, inj_params=inj_params,
                      deriv_symbs_string=deriv_symbs_string, conv_log=conv_log, use_rot=use_rot)
        nnet.setup_ant_pat_lpf_psds()
        nnet.calc_wf_polarizations()
        nnet.calc_snrs()
        if nnet.snr>8:
            isobservable[i] = True

    print('Unweighted number of these BBH mergers with single-detector SNR>8: %i' %np.sum(isobservable))
    print('Estimated number of these BBH mergers detectable over 1 year of observation time at LIGO design sensitivity')
    print('%i' %np.sum(weights[ofinterest][isobservable]))


##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":

    sample_detections()
