import numpy as np
import h5py
import pandas as pd

from gwosc import datasets
from gwpy.timeseries import TimeSeries
import bilby

import os
import sys
import argparse


def load_strain(event, det='H1'):
    gw_events = np.array(['GW241011','GW241110'])
    durations = [32., 16.]
    minimum_frequencies = [20., 30.]
    trigger_times = [1412725132.9554443, 1415277701.675049]
    if event not in gw_events:
        raise ValueError('Events must be part of %s' %gw_events)
    n = np.where(gw_events==event)[0][0]
    n_samples = 100
    minimum_frequency = minimum_frequencies[n]
    maximum_frequency = 1792.
    roll_off = 1.0 # Roll off duration of tukey window in seconds
    duration = durations[n]  # Analysis segment duration
    post_trigger_duration = 2  # Time between trigger time and end of segment
    trigger_time = trigger_times[n]

    PE_dir = '/home/tristan-bruel/Documents/Science/LVK/PE/%s' %event

    ## LVK strain data ##
    strain_filename = './waveforms/strain_%s_%s.txt' %(event,det)
    if not os.path.exists(strain_filename):
        #trigger_time = datasets.event_gps("GW241011_233834")
        #detectors = datasets.event_detectors("GW241011_233834")
        #if det not in detectors:
        #    raise ValueError('Detector %s not available for event %s' %(det,event))
        end_time = trigger_time + post_trigger_duration
        start_time = end_time - duration

        #psd_duration = 1024
        #psd_start_time = start_time - psd_duration
        #psd_end_time = start_time

        ifo = bilby.gw.detector.get_empty_interferometer(det)
        data = TimeSeries.fetch_open_data(det, start_time, end_time)
        data = data.resample(4096.)
        ifo.strain_data.set_from_gwpy_timeseries(data)

        #psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time, cache=True)
        #psd_data = psd_data.resample(4096)
        #psd_alpha = 2 * roll_off / duration
        #psd = psd_data.psd(
        #    fftlength=duration, overlap=0.5, window=("tukey", psd_alpha), method="median"
        #)
        psd_file = os.path.join(PE_dir, 'psd_%s.txt' %det)
        freqs, psd = np.loadtxt(psd_file).T
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=freqs, psd_array=psd
        )

        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency

        #times_det = ifo.time_array - ifo.strain_data.start_time
        times_det = ifo.time_array
        strain_det = ifo.whitened_time_domain_strain
        np.savetxt(strain_filename, 
                   np.array([times_det, strain_det]).T
                   )
    else:
        times_det, strain_det = np.loadtxt(strain_filename).T

    return times_det, strain_det 



def compute_waveforms(event, det):
    gw_events = np.array(['GW241011','GW241110'])
    durations = [32., 16.]
    minimum_frequencies = [20., 30.]
    if event not in gw_events:
        raise ValueError('Events must be part of %s' %gw_events)
    n = np.where(gw_events==event)[0][0]
    n_samples = 100
    minimum_frequency = minimum_frequencies[n]
    maximum_frequency = 1792.
    roll_off = 1.0 # Roll off duration of tukey window in seconds
    duration = durations[n] # Analysis segment duration
    post_trigger_duration = 2  # Time between trigger time and end of segment

    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo.duration = duration
    ifo.sampling_frequency = 4096.


    PE_dir = '/home/tristan-bruel/Documents/Science/LVK/PE/%s' %event

    ## Std LVK priors ##
    wvf_filename = './waveforms/waveforms_std_%s_%s.txt' %(event,det)
    if not os.path.exists(wvf_filename):
        ## Load PE samples from standard LVK priors ##
        '''
        file = os.path.join(PE_dir,'standard/web/samples/posterior_samples.h5')
        with h5py.File(file, 'r') as h:
            pos = h['C00:IMRPhenomXPHM']['posterior_samples']
            Mc_std = pos['chirp_mass']
            q_std = pos['mass_ratio']
            a1_std = pos['a_1']
            a2_std = pos['a_2']
            tilt1_std = pos['tilt_1']
            tilt2_std = pos['tilt_2']
            Dl_std = pos['luminosity_distance']
            z_std = pos['redshift']
            phi_12_std = pos['phi_12']
            phi_jl_std = pos['phi_jl']
            theta_jn_std = pos['theta_jn']
            phase_std = pos['phase']
            ra_std = pos['ra']
            dec_std = pos['dec']
            geocent_time_std = pos['geocent_time']
            psi_std = pos['psi']
            azimuth_std = pos['azimuth']
            zenith_std = pos['zenith']
        '''
        file = os.path.join(PE_dir,'standard/%s_merged_result.hdf5' %event)
        with h5py.File(file, 'r') as h:
            pos = h['posterior']
            Mc_std = pos['chirp_mass'][:]
            q_std = pos['mass_ratio'][:]
            a1_std = pos['a_1'][:]
            a2_std = pos['a_2'][:]
            tilt1_std = pos['tilt_1'][:]
            tilt2_std = pos['tilt_2'][:]
            Dl_std = pos['luminosity_distance'][:]
            z_std = pos['redshift'][:]
            phi_12_std = pos['phi_12'][:]
            phi_jl_std = pos['phi_jl'][:]
            theta_jn_std = pos['theta_jn'][:]
            phase_std = pos['phase'][:]
            ra_std = pos['ra'][:]
            dec_std = pos['dec'][:]
            geocent_time_std = pos['geocent_time'][:]
            psi_std = pos['psi'][:]
            azimuth_std = pos['azimuth'][:]
            zenith_std = pos['zenith'][:]

        ifo.start_time = np.mean(geocent_time_std) - duration + post_trigger_duration
        waveform_generator = bilby.gw.waveform_generator.LALCBCWaveformGenerator(
            duration=duration, sampling_frequency=4096,
            start_time=ifo.start_time,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            time_domain_source_model=None,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={'reference_frequency': 20.0,
                                'waveform_approximant': 'IMRPhenomXPHM',
                                'minimum_frequency': minimum_frequency,
                                'maximum_frequency': 1792,
                                'catch_waveform_errors': True,
                                'pn_spin_order': -1,
                                'pn_tidal_order': -1,
                                'pn_phase_order': -1,
                                'pn_amplitude_order': 0,
                                'mode_array': None}
        )

        idx_samples = np.random.choice(len(Mc_std), size=n_samples, replace=False)
        samples_std = {
                   'chirp_mass':Mc_std[idx_samples],
                   'mass_ratio':q_std[idx_samples],
                   'luminosity_distance':Dl_std[idx_samples],
                   'a_1':a1_std[idx_samples],
                   'a_2':a2_std[idx_samples],
                   'tilt_1':tilt1_std[idx_samples],
                   'tilt_2':tilt2_std[idx_samples],
                   'phi_12':phi_12_std[idx_samples],
                   'phi_jl':phi_jl_std[idx_samples],
                   'theta_jn':theta_jn_std[idx_samples],
                   'phase':phase_std[idx_samples],
                   'ra':ra_std[idx_samples],
                   'dec':dec_std[idx_samples],
                   'geocent_time':geocent_time_std[idx_samples],
                   'psi':psi_std[idx_samples],
                   'azimuth':azimuth_std[idx_samples],
                   'zenith':zenith_std[idx_samples],
             }
        samples_std = pd.DataFrame(samples_std)
        waveforms_std = list()
        for params in samples_std.to_dict(orient="records"):
            wf_pols = waveform_generator.frequency_domain_strain(params)
            fd_waveform = ifo.get_detector_response(wf_pols, params)
            whitened_fd_waveform = ifo.whiten_frequency_series(fd_waveform)
            td_waveform = ifo.get_whitened_time_series_from_whitened_frequency_series(
                whitened_fd_waveform
            )
            waveforms_std.append(td_waveform)
        waveforms_std = np.array(waveforms_std)
        wvf_mean = np.mean(waveforms_std, axis=0)
        wvf_low = np.percentile(waveforms_std, 5, axis=0)
        wvf_high = np.percentile(waveforms_std, 95, axis=0)
        header = 'First column is time, then 5 percentile, mean, and 95 percentile, over 100 waveform generations'
        np.savetxt(wvf_filename,
                   np.array([ifo.time_array, wvf_low, wvf_mean, wvf_high]).T,
                   )
    
    ## Astro priors ##
    wvf_filename = './waveforms/waveforms_astro_%s_%s.txt' %(event,det)
    if not os.path.exists(wvf_filename):    
        ## Load PE samples from astro priors ##
        '''
        file = os.path.join(PE_dir,'astro_pop_prior/web/samples/posterior_samples.h5')
        with h5py.File(file, 'r') as h:
            pos = h['C00:IMRPhenomXPHM']['posterior_samples']
            m1_astro = pos['mass_1_source']
            q_astro = pos['mass_ratio']
            Mc_astro = pos['chirp_mass']
            a1_astro = pos['a_1']
            a2_astro = pos['a_2']
            chi_eff_astro = pos['chi_eff']
            Dl_astro = pos['luminosity_distance']
            z_astro = pos['redshift']
        print('Number of PE samples in my standard analysis:', len(m1_astro))
        '''
        file = os.path.join(PE_dir,'astro_pop_prior/%s_astro_merged_result.hdf5' %event)
        with h5py.File(file, 'r') as h:
            pos = h['posterior']
            Mc_astro = pos['chirp_mass'][:]
            q_astro = pos['mass_ratio'][:]
            a1_astro = pos['a_1'][:]
            a2_astro = pos['a_2'][:]
            tilt1_astro = pos['tilt_1'][:]
            tilt2_astro = pos['tilt_2'][:]
            Dl_astro = pos['luminosity_distance'][:]
            z_astro = pos['redshift'][:]
            phi_12_astro = pos['phi_12'][:]
            phi_jl_astro = pos['phi_jl'][:]
            theta_jn_astro = pos['theta_jn'][:]
            phase_astro = pos['phase'][:]
            ra_astro = pos['ra'][:]
            dec_astro = pos['dec'][:]
            geocent_time_astro = pos['geocent_time'][:]
            psi_astro = pos['psi'][:]
            azimuth_astro = pos['azimuth'][:]
            zenith_astro = pos['zenith'][:]
        
        ifo.start_time = np.mean(geocent_time_astro) - duration + post_trigger_duration
        waveform_generator = bilby.gw.waveform_generator.LALCBCWaveformGenerator(
            duration=duration, sampling_frequency=4096,
            start_time=ifo.start_time,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            time_domain_source_model=None,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={'reference_frequency': 20.0,
                                'waveform_approximant': 'IMRPhenomXPHM',
                                'minimum_frequency': minimum_frequency,
                                'maximum_frequency': 1792,
                                'catch_waveform_errors': True,
                                'pn_spin_order': -1,
                                'pn_tidal_order': -1,
                                'pn_phase_order': -1,
                                'pn_amplitude_order': 0,
                                'mode_array': None}
        )


        idx_samples = np.random.choice(len(Mc_astro), size=n_samples, replace=False)
        samples_astro = {
                   'chirp_mass':Mc_astro[idx_samples],
                   'mass_ratio':q_astro[idx_samples],
                   'luminosity_distance':Dl_astro[idx_samples],
                   'a_1':a1_astro[idx_samples],
                   'a_2':a2_astro[idx_samples],
                   'tilt_1':tilt1_astro[idx_samples],
                   'tilt_2':tilt2_astro[idx_samples],
                   'phi_12':phi_12_astro[idx_samples],
                   'phi_jl':phi_jl_astro[idx_samples],
                   'theta_jn':theta_jn_astro[idx_samples],
                   'phase':phase_astro[idx_samples],
                   'ra':ra_astro[idx_samples],
                   'dec':dec_astro[idx_samples],
                   'geocent_time':geocent_time_astro[idx_samples],
                   'psi':psi_astro[idx_samples],
                   'azimuth':azimuth_astro[idx_samples],
                   'zenith':zenith_astro[idx_samples],
             }
        samples_astro = pd.DataFrame(samples_astro)
        waveforms_astro = list()
        for params in samples_astro.to_dict(orient="records"):
            wf_pols = waveform_generator.frequency_domain_strain(params)
            fd_waveform = ifo.get_detector_response(wf_pols, params)
            whitened_fd_waveform = ifo.whiten_frequency_series(fd_waveform)
            td_waveform = ifo.get_whitened_time_series_from_whitened_frequency_series(
                whitened_fd_waveform
            )
            waveforms_astro.append(td_waveform)
        waveforms_astro = np.array(waveforms_astro)
        wvf_mean = np.mean(waveforms_astro, axis=0)
        wvf_low = np.percentile(waveforms_astro, 5, axis=0)
        wvf_high = np.percentile(waveforms_astro, 95, axis=0)
        header = 'First column is time, then 5 percentile, mean, and 95 percentile, over 100 waveform generations'
        np.savetxt(wvf_filename,
                   np.array([ifo.time_array, wvf_low, wvf_mean, wvf_high]).T,
                   )



##################################################################
### Run it!
###
##################################################################
if __name__ == "__main__":
    

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", help="GW event to consider", default='GW241011')
    parser.add_argument("--det", help="Detector to use for the gw strain", default='H1')
    args = parser.parse_args()

    times_det, strain_det = load_strain(event=args.event, det=args.det)
    compute_waveforms(event=args.event, det=args.det)
