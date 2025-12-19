import linefinder.track as track

tracker = track.ParticleTracker(
    out_dir = '/scratch/tbruel/GBoF_III/m11e/data',
    tag = 'm11e_clustersofFIRE_pop1',
    sdir = '/net/cephfs/shares/feldmann.ics.mnf.uzh/data/GalaxiesOnFIRE/production_runs/metal_diffusion/m11e_res7100/output',
    p_types = [ 0, 4 ],
    snum_start = 598,
    snum_end = 600,
    snum_step = 1,
)


tracker.save_particle_tracks()
