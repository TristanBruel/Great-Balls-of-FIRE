import numpy as np
import h5py

import os
import argparse



def sampleBBHs(halo_idx, sample=None, Zsun=0.017):
    ### Sample BBHs in the halo identified by {halo_idx} in snapshot 1200
    redshifts, times = np.loadtxt("./snapshot_times.txt", usecols=[2,3]).T
    tmax = times.max()

    #SFR_file = './SFRs/Mstar90/FB15N1024_halo{}_SFR.txt'.format(halo_idx, halo_idx)
    SFR_file = './SFRs/FB15N1024_halo{}_SFR.txt'.format(halo_idx, halo_idx)
    times, Zmean, Zscatter, mform, sfr = np.loadtxt(SFR_file).T

    ## Load sampled clusters ##
    cluster_file = './clusters/sampledMassiveClusters_{}_halo{}.hdf5'.format(sample, halo_idx)
    if not os.path.exists(cluster_file):
        print('Clusters sampled in halo {} not found for method {}.'.format(halo_idx, sample))
        return
    with h5py.File(cluster_file, 'r') as f:
        cluster_mass = f['Mass'][()]
        cluster_radius = f['HalfMassRadius'][()]
        cluster_Z = f['Metallicity'][()]
        cluster_tform = f['FormationTime'][()]
    
    ## Load clusters that form the grid ##
    grid_file = os.path.join('./grid/gridClusters_withKremer.hdf5')
    with h5py.File(grid_file, 'r') as f:
        grid_ids = f['id'][()]
        grid_mass = f['Mass'][()]
        grid_radius = f['HalfMassRadius'][()]
        grid_Z = f['Metallicity'][()]
        grid_nBBH = f['nBBHs'][()]
    grid_ids = np.array(grid_ids, dtype='U10')
    #print('Number of clusters in the grid:', len(grid_ids))
    #print('Total number of BBH mergers:', np.sum(grid_nBBH))

    log_M = np.log10(grid_mass)
    log_r = np.log10(grid_radius)
    log_Z = np.log10(grid_Z /Zsun)
    """
    total_grid = 0
    neighbor = np.zeros_like(cluster_mass, dtype='int')
    for k in range(len(cluster_mass)):
        diff_M = log_M - np.log10(cluster_mass[k])
        diff_r = log_r - np.log10(cluster_radius[k])
        diff_Z = log_Z - np.log10(cluster_Z[k] /Zsun)
        distances = (diff_M)**2 + (diff_r)**2 + (diff_Z)**2
        #distances = (diff_M)**2 + (diff_r)**2
        closest = np.argmin(distances)
        neighbor[k] = closest
        total_grid += grid_nBBH[closest]
    print('Number of BBH mergers sampled from the grid in halo {}: {}'.format(halo_idx, total_grid))

    ## Load BBHs ##
    all_bbh_file = './grid/allBBHs.dat'
    bbh_ids = np.loadtxt(all_bbh_file, dtype='str', usecols=[0])
    dat = np.loadtxt(all_bbh_file, usecols=np.arange(1,12))
    out = np.zeros((total_grid,11))
    tforms = np.zeros(total_grid)
    cluster_id = np.empty(total_grid, dtype='<U10')
    start = 0
    for k in range(len(cluster_mass)):
        closest = neighbor[k]
        nbhs = grid_nBBH[closest]
        if nbhs>0:
            locs = np.where(bbh_ids == grid_ids[closest])[0]
            stop = start+nbhs
            out[start:stop,] = dat[locs]
            tforms[start:stop] = cluster_tform[k]
            cluster_id[start:stop] = grid_ids[closest]
            start = stop
    """
    all_bbh_file = './grid/allBBHs_withKremer.dat'
    bbh_ids = np.loadtxt(all_bbh_file, dtype='str', usecols=[0], skiprows=1)
    dat = np.loadtxt(all_bbh_file, usecols=np.arange(1,12), skiprows=1)

    #N_occ = np.zeros_like(grid_nBBH)

    total_grid = 0
    N_neighbors = 10
    out = np.zeros((0,dat.shape[1]))
    tforms = []
    cluster_id = []
    cluster_metals = []
    list_bbhs = np.array([np.where(bbh_ids == i)[0] for i in grid_ids], dtype='object')
    for k in range(len(cluster_mass)):
        diff_M = log_M - np.log10(cluster_mass[k])
        diff_r = log_r - np.log10(cluster_radius[k])
        diff_Z = log_Z - np.log10(cluster_Z[k] /Zsun)
        distances = (diff_M)**2 + (diff_r)**2 + (diff_Z)**2

        #closest = np.argmin(distances)
        #N_occ[closest] += 1

        neighbors = np.argpartition(distances, N_neighbors)[:N_neighbors]
        n_mean = round(np.average(grid_nBBH[neighbors], weights=1/distances[neighbors]))
        if n_mean>0:
            candidates = np.concatenate(list_bbhs[neighbors])
            weights = np.repeat(1/distances[neighbors], grid_nBBH[neighbors])
            locs = np.random.choice(candidates, size=n_mean, replace=False, p=weights/np.sum(weights))
            out = np.vstack([out, dat[locs]])
            tforms.extend([cluster_tform[k]]*n_mean)
            cluster_id.extend(bbh_ids[locs])
            cluster_metals.extend([cluster_Z[k]]*n_mean)
            total_grid += n_mean
    tforms = np.array(tforms)
    cluster_id = np.array(cluster_id)
    cluster_metals = np.array(cluster_metals)

    #np.savetxt('occurences_{}_halo{}.dat'.format(sample,halo_idx), N_occ)

    if total_grid==0:
        #print('No BBH merger in halo {}'.format(halo_idx))
        return

    ## Associate each grid bbh with the formation time of the sampled cluster, and recompute the time of merger ##
    dt_merge = out[:,6] - out[:,1]
    tmerger = tforms+dt_merge
    out[:,1] = tforms
    out[:,6] = tmerger
    
    output_file = './BBHs/sampledBBHs_{}_halo{}_Kremer.dat'.format(sample,halo_idx)
    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as output:
        with open(all_bbh_file, 'r') as bbhs:
            # header
            header = bbhs.readline()
            h = header.split()
            h.insert(1, '#2:Metal')
            h[0] = '#0:ID'
            h[1] = '#1:Ztrue'
            header = ' '.join(h)
            output.write(header)
        for k in range(total_grid):
            line = cluster_id[k] + ' {:.8e}'.format(cluster_metals[k]) + str(out[k,:]).replace('[',' ').replace(']','').replace('\n','') +'\n'
            output.write(line)

    bbhs = {'mBH1': out[:,3], 'mBH2': out[:,4], 'tmerge': out[:,6]}
    return bbhs


##################################################################
### 
### Run it!
###
##################################################################
if __name__ == "__main__":

    #####################################
    # Define command line options
    parser = argparse.ArgumentParser()
    parser.add_argument('--halo', dest='halo_idx', type=int, default=20, help='Halo index at redshift 0.')
    parser.add_argument('--sample', dest='sample', type=str, help='Method to use for the sampling of massive clusters.')
    args = parser.parse_args()

    #####################################
    # Plot 
    sampleBBHs(halo_idx=args.halo_idx, sample=args.sample)
