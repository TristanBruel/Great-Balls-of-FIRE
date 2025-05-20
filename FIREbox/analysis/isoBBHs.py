import pandas as pd
import numpy as np
import h5py
from scipy.interpolate import interp1d

import os
from glob import glob
import time
import argparse

import gizmo_analysis as gizmo
import utilities as ut
import auxiliary_SFR as aux_SFR


def linear(x, a, b):
    return a*x + b


def isolatedBBHs(COSMIC_path='./', sample='SFR'):

    ### Load the COSMIC part ###
    BBHs_path = os.path.join(COSMIC_path, 'COSMIC_BBHs.h5')
    COSMIC_BBHs = pd.read_hdf(BBHs_path, header=None)
    eta_file = './etas_bbhs.dat'
    COSMIC_metallicities, etas = np.loadtxt(eta_file, usecols=[0,3]).T
    n_COSMIC_metallicities = len(COSMIC_metallicities)

    extrapol = interp1d(x=np.log(COSMIC_metallicities), y=np.log(etas), kind='linear', fill_value='extrapolate')
    
    ### Load the FIREbox part ###
    redshifts, times = np.loadtxt("/data/tbruel/FIREbox/snapshot_times.txt", usecols=[2,3]).T
    ## Load halos at z=0 ##
    halopath = '/net/cephfs/shares/feldmann.ics.mnf.uzh/data/FIREbox/analysis/AHF/FB15N1024/halo/1200'
    halofile = glob(halopath+'/*.AHF_halos')[0]
    halos = pd.read_csv(halofile, sep='\t')
    '''
    # Use Elia's galaxies
    halofile = '/scratch/tbruel/snap1200_circularity_M8_sample.hdf5'
    galaxies = pd.read_hdf(halofile)
    halo_ids = np.array(galaxies['halo ID'])
    '''
    # Load star particles and properties
    snap_dir = '/net/cephfs/shares/feldmann.ics.mnf.uzh/data/FIREbox/production_runs/FB15N1024'
    part = gizmo.io.Read.read_snapshots(['star'], 'redshift', 0,
            simulation_directory='/data/tbruel/FIREbox',
            snapshot_directory=snap_dir,
            )
    h = part.Cosmology['hubble']
    t0 = part.snapshot['time']
    star_particles = part['star']
    positions = star_particles['position']
    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]

    # Select halos that are not subhalos and that are more massive than ~ 10^7 Msun
    Mstar_file = 'Mstar90.dat'
    halo_ids, Mstar, Rgal = np.loadtxt(Mstar_file, skiprows=1, usecols=[0,3,4]).T
    halo_ids = halo_ids.astype(np.int64)

    ## Load GCMF slope from my cosmo simulations ##
    #alpha_fit = np.loadtxt('./fit_alpha_stellarmass')[:2]
    #alpha_stdev = np.loadtxt('./fit_alpha_stellarmass')[2]

    ### Sample BBHs ###
    print('Starting sampling...')
    start = time.time()

    for n,halo_idx in enumerate(halo_ids):
        galaxy = halos.loc[halo_idx]
        print('Halo:', halo_idx)
        Ms = Mstar[n]
        Xhalo, Yhalo, Zhalo = galaxy[['Xc(6)', 'Yc(7)', 'Zc(8)']] /h
        Mvir = galaxy['Mvir(4)'] /h
        Rvir = galaxy['Rvir(12)'] /h

        in_gal = np.sqrt((x-Xhalo)**2+(y-Yhalo)**2+(z-Zhalo)**2) < Rgal[n]
        metallicities = star_particles.prop('massfraction.metals')[in_gal]
        form_time = star_particles.prop('form.time')[in_gal] # formation times in Gyr
        #form_mass = star_particles.prop('form.mass')[in_gal]
        age = (t0 - form_time) *1e9 # age in yr
        form_mass = star_particles['mass'][in_gal] *aux_SFR.mass_ratio(age, metallicities)
        closest_COSMIC_Zidx = np.array([(np.abs(COSMIC_metallicities-Z)).argmin() for Z in metallicities])

        SFR_file = './SFRs/Rvir/FB15N1024_halo{}_SFR.txt'.format(halo_idx)
        if not os.path.exists(SFR_file):
            print('No SFR file found for this halo.')
            print(SFR_file)
            return
        else:
            times, Zmean, Zscatter, mform, sfr = np.loadtxt(SFR_file).T

        clusterfile = './clusters/sampledMassiveClusters_{}_halo{}.hdf5'.format(sample,halo_idx)
        with h5py.File(clusterfile, 'r') as clusters:
            tform = clusters['FormationTime'][()]
            Mcl = clusters['Mass'][()]
        cluster_mform, _ = np.histogram(tform, bins=times, weights=Mcl)
        ## Rescale to get an estimate mass of ALL star clusters, not just the massive ones ##
        #alpha0 = linear(np.log10(Ms), *alpha_fit)
        #alpha_est = np.random.normal(loc=alpha0, scale=alpha_stdev)
        #alpha_est = max(-2,alpha_est)
        #cluster_mform *= np.power(6e4/1e3, alpha_est+2) # estimate of total mass of clusters, not just the massive ones
        alpha_est = -2
        cluster_mform *= (np.log(1e8)-np.log(1e3)) /(np.log(1e8)-np.log(6e4))

        sfr_clusters = cluster_mform /np.diff(times *1e9) # sfr in unit Msun yr^-1
        sfr_clusters = np.append(0, sfr_clusters)

        temp_file = './BBHs/isolated/halo{}_BBHs.csv'.format(halo_idx)
        for Zidx in range(n_COSMIC_metallicities):
            # load the BBHs from COSMIC associated to metallicity Zidx
            BBHs_Z = COSMIC_BBHs[COSMIC_BBHs.COSMIC_metallicity == COSMIC_metallicities[Zidx]]

            # filter the star particles with metallicity closest to Zidx
            mask_Z = closest_COSMIC_Zidx==Zidx
            form_mass_Z = form_mass[mask_Z]
            form_time_Z = form_time[mask_Z]
            metallicities_Z = metallicities[mask_Z]
            print('Number of star particles around metallicity Z{}: {}'.format(Zidx+1, len(form_mass_Z)))
            print('Total mass of star particles: {:.3e} Msun'.format(np.sum(form_mass_Z)))
            '''
            if len(form_mass_Z) >0:
                # Reweight with SFR from isolated binaries (\clusters)
                snapshots_Z = np.array([np.where(t < times)[0][0] for t in form_time_Z])
                SFRw = max(sfr[snapshots_Z]-sfr_clusters[snapshots_Z],0) /sfr[snapshots_Z]
                # Reweight with BBH formation efficiency
                etas = np.exp(extrapol(np.log(metallicities_Z)))

                weights = form_mass_Z *etas *SFRw # average 'number' of BBHs formed

                pop = BBHs_Z.sample(len(form_mass_Z), replace=True)
                pop['tbirth'] = form_time_Z
                pop['star_Z'] = metallicities_Z
                pop['weight'] = weights
                pop.to_csv(temp_file, mode='a', header=False, index=False)
                print('Number of BBHs sampled: ', np.sum(weights))
            '''
            BBH_count_Z = 0
            t_birth_Z = [] # stores the epoch of formation of each star particle that will produce a BBH merger
            star_metallicity = [] # stores the metallicity of each star particle that will produce a BBH merger
            for s in range(len(form_mass_Z)):
                eta = np.exp(extrapol(np.log(metallicities_Z[s]))) # formation efficiency
                snap = np.where(form_time_Z[s] < times)[0][0]
                if sfr[snap] > 0:
                    SFRw = max(sfr[snap]-sfr_clusters[snap],0) /sfr[snap]
                else:
                    SFRw = 0
                N_BBH = form_mass_Z[s] *eta *SFRw # estimated 'number' of BBHs formed
                if N_BBH>1:
                    Bernoulli_draw = np.random.binomial(1, N_BBH%1)
                    count = int(N_BBH) + Bernoulli_draw
                    t_birth_Z.extend([form_time_Z[s]] *count)
                    star_metallicity.extend([metallicities_Z[s]] *count)
                    BBH_count_Z += count
                elif N_BBH<0:
                    print(halo_idx)
                    print(snap)
                else:
                    Bernoulli_draw = np.random.binomial(1, N_BBH)
                    if Bernoulli_draw>0:
                        t_birth_Z.append(form_time_Z[s])
                        star_metallicity.append(metallicities_Z[s])
                        BBH_count_Z += 1

            if BBH_count_Z > 0:
                pop = BBHs_Z.sample(BBH_count_Z, replace=True)
                pop['tbirth'] = t_birth_Z
                pop['star_Z'] = star_metallicity
                pop.to_csv(temp_file, mode='a', header=False, index=False)
                print('Number of BBHs sampled: ', BBH_count_Z)

        iso_BBH_pop = pd.read_csv(temp_file, header=None)
        iso_BBH_pop.columns = pop.columns
        output_file = './BBHs/isolated/halo{}_BBHs.h5'.format(halo_idx)
        iso_BBH_pop.to_hdf(output_file, key='FIRE_pop')
        os.remove(temp_file)

    print('Sampling completed !')
    end = time.time()
    print('Time elapsed: {:.3f}s'.format(end-start))
    
    return iso_BBH_pop


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Run it
    isolatedBBHs(COSMIC_path='./')
