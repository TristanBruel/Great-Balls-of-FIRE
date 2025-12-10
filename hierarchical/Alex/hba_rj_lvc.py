import os

os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)

import numpy as np
#import draw_pop_flex as draw_pop
from scipy.interpolate import RectBivariateSpline as spline2d
import matplotlib.pyplot as plt

import scipy.integrate

from scipy.interpolate import InterpolatedUnivariateSpline as spline

import scipy.stats as stats

# from bilby.core.result import read_in_result

from eryn.ensemble import EnsembleSampler
from eryn.state import State
# from eryn.prior import PriorContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove
from eryn.moves.tempering import make_ladder
from scipy.interpolate import RegularGridInterpolator as splinend
from eryn.prior import ProbDistContainer, uniform_dist

import pdfs as pdfs_func

from astropy.cosmology import Planck15
import astropy.units as u


import corner as triangle

import time

import argparse
try:
    import cupy as xp
    xp.cuda.runtime.setDevice(6)
except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    print('use cpu')



import itertools

def selection_function(params,groups):

    xgauss,values_pl,values_global,values_spins,xqpl,xz = params
    group_gauss,group_pl,group_global,group_spins,group_qpl,group_z = groups


    ampl_qpl=None
    betaq=np.copy(xqpl[:,0])
    group_qgauss=None
    norm_m1=False
    norm_m1_num=False
    ampl_qbpl=None
    xqbpl=None
    group_qbpl=None
    xqgauss=None
    xneg_gauss=None
    group_neg_gauss=None
    xbpl=None
    group_bpl=None
    return_norm=False

    if args.only_gauss==0:
        pdf_m1=pdfs_func.m1_pdf(injections['m1'],xgauss,group_gauss,values_pl,group_pl,values_global,group_global,xneg_gauss=xneg_gauss,group_neg_gauss=group_neg_gauss,neg_gauss=args.neg_gauss,xbpl=xbpl,group_bpl=group_bpl,broken_pl=args.broken_pl,norm=norm_m1,norm_num=norm_m1_num,return_norm=return_norm,nsmooth=args.smooth)

    else:
        pdf_m1=pdfs_func.m1_pdf_gauss(injections['m1'],xgauss,group_gauss,values_global,group_global,xneg_gauss=xneg_gauss,group_neg_gauss=group_neg_gauss,neg_gauss=args.neg_gauss,return_norm=return_norm,to_norm=norm_m1)
        

    pdf_q=pdfs_func.q_pdf_sharp(injections['q'],injections['m1'],betaq,values_global)

    pdf_tilt=pdfs_func.spin_tilt_pdf(injections['tilt1'],injections['tilt2'],values_spins)

    pdf_chi1=pdfs_func.spin_mag_pdf(injections['chi1'],values_spins)
    pdf_chi2=pdfs_func.spin_mag_pdf(injections['chi2'],values_spins)

    pdf_z=pdf_z=pdfs_func.z_pdf(injections['z'],xp.asarray(xz[:,0]),dVc_spl,zmax)

    pdf=pdf_m1*pdf_q*pdf_tilt*pdf_chi1*pdf_chi2*pdf_z

    mu=(1./injections['ninjs'])*xp.sum(pdf/injections['priors'][None,:],axis=1)
    var=(1./injections['ninjs']**2)*xp.sum((pdf/injections['priors'][None,:])**2,axis=1)-mu**2/injections['ninjs']

    Neff=mu**2/var

    ans=mu/np.exp((3.+nevents)/(2.*Neff))

    ans[Neff<4*nevents]=xp.inf

    ans[xp.sum(pdf_m1<0,axis=1)>0]=0
    ans[xp.sum(xp.isinf(pdf_m1),axis=1)>0]=0
    ans[xp.sum(xp.isinf(pdf_q),axis=1)>0]=0
    ans[xp.sum(pdf_q<0,axis=1)>0]=0

    return ans



def loglike(xs,groups):

   
    xgauss0,xpl,xglobal,xspins,xqpl,xz = xs
    group_gauss,group_pl,group_global,group_spins,group_qpl,group_z = groups

    xgauss=xp.copy(xp.asarray(xgauss0))
    if args.flat_ampl==1:
        # breakpoint()
        xgauss[:,0]=xp.log10(xp.asarray(xgauss[:,0]))

        

    values_pl={}
    for kpar,par in enumerate(params_vary['pl']):
        values_pl[par]=xpl[:,kpar]

    if args.flat_ampl==1:
        values_pl['log10_lambda_pl']=xp.log10(xp.asarray(values_pl['lambda_pl']))
   
    values_global={}
    for kpar,par in enumerate(params_vary['global']): 
        values_global[par]=xglobal[:,kpar]
       

    values_spins={}
    for kpar,par in enumerate(params_vary['spins']):   
        values_spins[par]=xspins[:,kpar]

    if args.sigma2==1:
        values_spins['varchi']=np.sqrt(values_spins['varchi'])
      

    values_spins['abeta']=(values_spins['meanchi']-values_spins['meanchi']**2-values_spins['varchi']**2)*(values_spins['meanchi']/values_spins['varchi']**2)
    values_spins['bbeta']=(values_spins['meanchi']-values_spins['meanchi']**2-values_spins['varchi']**2)*((1-values_spins['meanchi'])/values_spins['varchi']**2)

    # inds_neg=xp.argwhere(values_global['mmin']<0)
    # 200 ms

    ampl_qpl=None
    betaq=np.copy(xqpl[:,0])
    group_qgauss=None
    norm_m1=False
    norm_m1_num=False
    ampl_qbpl=None
    xqbpl=None
    group_qbpl=None
    xqgauss=None
    xneg_gauss=None
    group_neg_gauss=None
    xbpl=None
    group_bpl=None
    return_norm=False

    if args.only_gauss==0:
        pdf_m1=pdfs_func.m1_pdf(data['m1'],xgauss,group_gauss,values_pl,group_pl,values_global,group_global,xneg_gauss=xneg_gauss,group_neg_gauss=group_neg_gauss,neg_gauss=args.neg_gauss,xbpl=xbpl,group_bpl=group_bpl,broken_pl=args.broken_pl,norm=norm_m1,norm_num=norm_m1_num,return_norm=return_norm,nsmooth=args.smooth)

    else:
        pdf_m1=pdfs_func.m1_pdf_gauss(data['m1'],xgauss,group_gauss,values_global,group_global,xneg_gauss=xneg_gauss,group_neg_gauss=group_neg_gauss,neg_gauss=args.neg_gauss,return_norm=return_norm,to_norm=norm_m1)
        
    
    pdf_q=pdfs_func.q_pdf_sharp(data['q'],data['m1'],betaq,values_global)

    pdf_tilt=pdfs_func.spin_tilt_pdf(data['tilt1'],data['tilt2'],values_spins)

    pdf_chi1=pdfs_func.spin_mag_pdf(data['chi1'],values_spins)
    pdf_chi2=pdfs_func.spin_mag_pdf(data['chi2'],values_spins)

   
    pdf_z=pdf_z=pdfs_func.z_pdf(data['z'],xp.asarray(xz[:,0]),dVc_spl,zmax)


    # ~0 ms
    pdf=pdf_m1*pdf_q*pdf_tilt*pdf_chi1*pdf_chi2*pdf_z
    integrand=(pdf/(data['priors'][None,:])).reshape(-1,nevents,nsamp)

    bfs=xp.sum(integrand,axis=2)

    if args.neff_min==1:
        neffs=bfs**2/xp.sum(integrand**2,axis=2)
        bfs[neffs<nevents]=0

    # ~0 ms
    logl=xp.sum(xp.log((1./nsamp)*bfs),axis=1)
    first_logl = logl.copy()

    selection_function_term=selection_function([xgauss,values_pl,values_global,values_spins,xqpl,xz],[group_gauss,group_pl,group_global,group_spins,group_qpl,group_z])
    
    logl+=-selection_function_term

    logl[xp.isinf(first_logl)]=-1E300

    logl[xp.isinf(logl)]=-1E300

    logl[selection_function_term==0]=-1E300

    # logl[xp.isinf(first_logl)] = -1e300

    groups_to_concatenate=[group_gauss]
    if args.only_gauss==0:
        groups_to_concatenate+=[group_pl]
   
    allgroups=np.unique(np.concatenate(groups_to_concatenate))

    indsabsent=np.setdiff1d(np.arange(len(xglobal)),allgroups)

    logl[indsabsent]=-1E300

    logl[xp.isinf(logl)]=-1E300

    logl[values_spins['abeta']<0]=-1E300
    logl[values_spins['bbeta']<0]=-1E300

    if xp.sum(xp.isnan(logl)):
        breakpoint()
    try:
        logl = logl.get()
    except AttributeError:
        pass
    return logl

def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='script_snr.py',description='compute LISA SNRs of LVC events')
    # parser.add_argument('-ev', '--event', default='GW190521')
    parser.add_argument('-id', '--id',type=int,default=1)
    parser.add_argument('-large_sigma', '--large_sigma',type=int,default=0)
    parser.add_argument('-long_burn', '--long_burn',type=int,default=0)
    parser.add_argument('-long_run', '--long_run',type=int,default=0)
    parser.add_argument('-neff_min', '--neff_min',type=int,default=1)
    parser.add_argument('-red_prior', '--red_prior',type=int,default=2)
    parser.add_argument('-ngauss_max', '--ngauss_max',type=int,default=1)
    parser.add_argument('-ngauss_neg_max', '--ngauss_neg_max',type=int,default=None)
    parser.add_argument('-ngauss_min', '--ngauss_min',type=int,default=1)
    parser.add_argument('-only_gauss', '--only_gauss',type=int,default=0)
    parser.add_argument('-neg_gauss', '--neg_gauss',type=int,default=0)
    parser.add_argument('-broken_pl', '--broken_pl',type=int,default=0)
    parser.add_argument('-broken_pl_q', '--broken_pl_q',type=int,default=0)
    parser.add_argument('-zero_pl', '--zero_pl',type=int,default=0)
    parser.add_argument('-no_pl', '--no_pl',type=int,default=0)
    parser.add_argument('-test', '--test',type=int,default=0)
    parser.add_argument('-flat_ampl', '--flat_ampl',type=int,default=1)
    parser.add_argument('-high_ampl', '--high_ampl',type=int,default=0)
    parser.add_argument('-sigma2', '--sigma2',type=int,default=0)
    parser.add_argument('-lvc_data', '--lvc_data',type=int,default=0)
    parser.add_argument('-lvc_samples', '--lvc_samples',type=int,default=1)
    parser.add_argument('-new_data', '--new_data',type=int,default=0)
    parser.add_argument('-zmax', '--zmax',type=float,default=2.3)
    parser.add_argument('-lvc_prior', '--lvc_prior',type=int,default=1)
    parser.add_argument('-small_jump', '--small_jump',type=int,default=0)
    parser.add_argument('-small_sigma', '--small_sigma',type=int,default=0)
    parser.add_argument('-small_mu', '--small_mu',type=int,default=0)
    parser.add_argument('-large_mu', '--large_mu',type=int,default=0)
    parser.add_argument('-gaussian_q', '--gaussian_q',type=int,default=0)
    parser.add_argument('-smooth', '--smooth',type=int,default=0)
    parser.add_argument('-broad_kappa', '--broad_kappa',type=int,default=0)
    parser.add_argument('-fix_qpl', '--fix_qpl',type=int,default=0)
    parser.add_argument('-less_walkers', '--less_walkers',type=int,default=0)
    parser.add_argument('-no_q_smooth', '--no_q_smooth',type=int,default=0)
    parser.add_argument('-norm_num', '--norm_num',type=int,default=1)
    parser.add_argument('-joint_m1z', '--joint_m1z',type=int,default=0)
    parser.add_argument('-prior_move', '--prior_move',type=int,default=0)
    parser.add_argument('-no_qpl', '--no_qpl',type=int,default=0)
    parser.add_argument('-no_order_betaq', '--no_order_betaq',type=int,default=0)
    parser.add_argument('-no_order_alpha', '--no_order_alpha',type=int,default=0)
    parser.add_argument('-flat_leaves_gauss', '--flat_leaves_gauss',type=int,default=1)
    parser.add_argument('-wait', '--wait',type=int,default=0)
    parser.add_argument('-with_rate', '--with_rate',type=int,default=1)
    parser.add_argument('-o3_only', '--o3_only',type=int,default=0)
    parser.add_argument('-on_pc', '--on_pc',type=int,default=0)
    # parser.add_argument('-large_prior', '--large_prior',type=float,default=None)



    return parser.parse_args()

args=parser()

xp.random.seed(args.id)

if args.o3_only==0:
    file_injection=np.load('/data/atoubiana/ligo_pop_obs/selection_function_elements.npz')
else:
    file_injection=np.load('/data/atoubiana/ligo_pop_obs/selection_function_elements_o3.npz')
# file_injection=np.load('../population_ligo/selection_function_elements.npz')
injections={}
injections['m1']=xp.asarray(file_injection['m1s'])
injections['q']=xp.asarray(file_injection['qs'])
injections['z']=xp.asarray(file_injection['zs'])
injections['chi1']=xp.asarray(file_injection['chi1s'])
injections['chi2']=xp.asarray(file_injection['chi2s'])
injections['tilt1']=xp.asarray(file_injection['tilt1s'])
injections['tilt2']=xp.asarray(file_injection['tilt2s'])
injections['priors']=xp.asarray(file_injection['inj_priors'])
injections['ninjs']=xp.asarray(file_injection['ninjs'][0])

if args.o3_only==0:
    if args.on_pc==0:
        data_lvc=np.load('/data/atoubiana/ligo_pop_obs/lvc_data_lvc_samples_full.npz')
    else:
        data_lvc=np.load('lvc_data_lvc_samples_full.npz')
else:
    if args.on_pc==0:
        data_lvc=np.load('/data/atoubiana/ligo_pop_obs/lvc_data_lvc_samples_2.npz')
    else:
        data_lvc=np.load('lvc_data_lvc_samples_2.npz')
nevents=data_lvc['nevents'][0]
nsamp=data_lvc['nsamples'][0]
print(nevents,nsamp)
data={}
data['m1']=xp.asarray(data_lvc['m1s'].reshape(-1))
data['q']=xp.asarray(data_lvc['qs'].reshape(-1))
data['z']=xp.asarray(data_lvc['zs'].reshape(-1))
data['chi1']=xp.asarray(data_lvc['chi1s'].reshape(-1))
data['chi2']=xp.asarray(data_lvc['chi2s'].reshape(-1))
data['tilt1']=xp.asarray(data_lvc['tilt1s'].reshape(-1))
data['tilt2']=xp.asarray(data_lvc['tilt2s'].reshape(-1))
data['priors']=xp.asarray(data_lvc['priors'].reshape(-1))



params_vary={}
params_vary['pl']=['log10_lambda_pl','alpha']
params_vary['gauss']=['log10_lambda_peak','mum','sigmam']
params_vary['global']=['mmin','mmax','deltam']
params_vary['spins']=['zeta','sigmat','meanchi','varchi']
params_vary['q']=['betaq']
params_vary['z']=['kappa']

if args.flat_ampl==1:
    params_vary['pl'][0]='lambda_pl'
    params_vary['gauss'][0]='lambda_peak'

dict_dims={}
dict_nleaves_min={}
dict_nleaves_max={}

dict_nleaves_min['pl'],dict_nleaves_max['pl']=1,1
dict_nleaves_min['gauss'],dict_nleaves_max['gauss']=1,1
dict_nleaves_min['global'],dict_nleaves_max['global']=1,1
dict_nleaves_min['spins'],dict_nleaves_max['spins']=1,1
dict_nleaves_min['q'],dict_nleaves_max['q']=1,1
dict_nleaves_min['z'],dict_nleaves_max['z']=1,1



mins={}
maxs={}

if args.red_prior==0:
    mins['mum'],maxs['mum']=5,90
    mins['sigmam'],maxs['sigmam']=1,20
elif args.red_prior>0:
    mins['mum'],maxs['mum']=20,50
    mins['sigmam'],maxs['sigmam']=1,10

if args.small_mu==1:
    mins['mum']=2

if args.large_mu==1:
    maxs['mum']=100


if args.small_sigma==1:
    mins['sigmam']=0.1
    

if args.large_sigma==1:
    maxs['sigmam']=20

if args.large_sigma==2:
    maxs['sigmam']=30

if args.with_rate==1:
    mins['log10_lambda_peak'],maxs['log10_lambda_peak']=2.,10.
    mins['lambda_peak'],maxs['lambda_peak']=0,1e10

mins['alpha'],maxs['alpha']=1.1,10
mins['betaq'],maxs['betaq']=-1,7
mins['mmin'],maxs['mmin']=2,10
mins['mmax'],maxs['mmax']=30,100

mins['deltam'],maxs['deltam']=0.5,10

mins['log10_lambda_pl'],maxs['log10_lambda_pl']=-4,0

mins['lambda_pl'],maxs['lambda_pl']=0,1

if args.with_rate==1:
    mins['log10_lambda_pl'],maxs['log10_lambda_pl']=2,10
    
    mins['lambda_pl'],maxs['lambda_pl']=0,1e10


mins['zeta'],maxs['zeta']=0,1
mins['meanchi'],maxs['meanchi']=0,0.95
if args.sigma2==0:
    if args.red_prior>=2:
        mins['varchi'],maxs['varchi']=np.sqrt(0.005),0.5
    else:
        mins['varchi'],maxs['varchi']=0.01,0.5
elif args.sigma2==1:
    mins['varchi'],maxs['varchi']=0.005,0.25
mins['sigmat'],maxs['sigmat']=0.1,4


mins['kappa'],maxs['kappa']=-1,6

if args.broad_kappa==1:
    mins['kappa'],maxs['kappa']=-3,9

ngauss_max=args.ngauss_max
ngauss_min=args.ngauss_min




if args.zmax is None:
    zmin,zmax=0,5
else:
    zmin,zmax=0,args.zmax

zs=xp.linspace(zmin,zmax,1000)

try:
    dVc=xp.copy(Planck15.differential_comoving_volume(zs.get()))*4.*np.pi
    dVc_spl=spline(zs.get(),dVc)
except AttributeError:
    dVc=xp.copy(Planck15.differential_comoving_volume(zs))*4.*np.pi
    dVc_spl=spline(zs,dVc)



# dict_inds_vary_pl={}
# dict_inds_vary_global={}
# dict_inds_vary_spins={}
# if args.no_pl==0:
#     for kpar,par in enumerate(params_vary_pl):
#         dict_inds_vary_pl[par]=kpar
# for kpar,par in enumerate(params_vary_global):
#     dict_inds_vary_global[par]=kpar
# for kpar,par in enumerate(params_vary_spins):
#     dict_inds_vary_spins[par]=kpar

branch_names=['gauss','pl','global','spins','q','z']


for branch in branch_names:
    dict_dims[branch]=len(params_vary[branch])

ndims=[]
nleaves_min=[]
nleaves_max=[]
for branch in branch_names:
    ndims+=[dict_dims[branch]]
    nleaves_min+=[dict_nleaves_min[branch]]
    nleaves_max+=[dict_nleaves_max[branch]]

priors={}
for branch in branch_names:
    dict_prior_branch={}
    for kpar,par in enumerate(params_vary[branch]):
        dict_prior_branch[kpar]=uniform_dist(mins[par],maxs[par])
    priors[branch]=ProbDistContainer(dict_prior_branch)

nwalkers=50
ntemps=5
tempering_kwargs=dict(ntemps=ntemps)
betas = np.linspace(1.0, 0.0, ntemps)

if args.test==1:
    nsteps = 20
else:
    nsteps=2000
if args.long_run==1:
    nsteps=5000
# burn for 1000 steps
if args.test==1:
    burn = 5
else:
    burn=100
if args.long_burn==1:
    burn=1000
if args.long_burn==2:
    burn=45000
if args.long_burn==3:
    burn=70000
# thin by 5
thin_by = 1


create_path=True
path='/data/atoubiana/ligo_pop_obs/samples_full_selection_rj_lvc_%devs'%(nevents)

if args.o3_only==1:
    path+='_o3_only'

if args.with_rate==1:
    path+='_with_rate'

if args.smooth>0:
    path+='_smooth%d'%args.smooth

if args.no_q_smooth==1:
    path+='_no_q_smooth'

if args.joint_m1z==1:
    path+='_joint_m1z'

if args.joint_m1z==2:
    path+='_joint_m1z2'

if args.joint_m1z==3:
    path+='_joint_m1z3'

if args.joint_m1z==4:
    path+='_joint_m1z4'

if args.joint_m1z==5:
    path+='_joint_m1z5'

if args.zero_pl==1:
    path+='_zero_pl'

if args.no_pl==1:
    path+='_no_pl'

if args.only_gauss==1:
    path+='_only_gauss'

if args.broken_pl==1:
    path+='_broken_pl'

if args.no_order_alpha==1:
    path+='_no_order_alpha'

if args.broken_pl_q==1:
    path+='_broken_pl_q'

if args.broken_pl_q==2:
    path+='_broken_pl_q2'

if args.no_order_betaq==1:
    path+='_no_order_betaq'

if args.gaussian_q==1:
    path+='_gaussian_q'

if args.gaussian_q==2:
    path+='_gaussian_q2'

if args.fix_qpl==1:
    path+='_fix_qpl'

if args.no_qpl==1:
    path+='_no_qpl'

if args.high_ampl==1:
    path+='_high_ampl'

if args.flat_ampl==1:
    path+='_flat_ampl'

if args.sigma2==1:
    path+='_sigma2'

if not (ngauss_min==1 and ngauss_max==1):
    path+='_%d_gauss_min'%ngauss_min
    path+='_%d_gauss_max'%ngauss_max


if args.large_sigma==1:
    path+='_large_sigma'

if args.large_sigma==2:
    path+='_large_sigma2'

if args.long_run==1:
    path+='_long_run'

if args.long_run==2:
    path+='_long_run2'

if args.long_run==3:
    path+='_long_run3'

if args.long_run==4:
    path+='_long_run4'

if args.long_burn==1:
    path+='_long_burn'

if args.long_burn==2:
    path+='_long_burn2'

if args.long_burn==3:
    path+='_long_burn3'

if args.long_burn==4:
    path+='_long_burn4'

if args.long_burn==5:
    path+='_long_burn5'

if args.neff_min==1:
    path+='_neff_min'

if args.red_prior==1:
    path+='_red_prior'

if args.red_prior==2:
    path+='_red_prior2'

if args.red_prior==3:
    path+='_red_prior3'

if args.small_sigma==1:
    path+='_small_sigma'

if args.small_mu==1:
    path+='_small_mu'

if args.large_mu==1:
    path+='_large_mu'

if args.broad_kappa==1:
    path+='_broad_kappa'

if args.small_jump==1:
    path+='_small_jump'

if args.prior_move==1:
    path+='_prior_move'

if args.zmax is not None:
    path+='_zmax_%1.1f'%args.zmax

if args.less_walkers==1:
    path+='_less_walkers'

if args.less_walkers==2:
    path+='_less_walkers2'

if args.test==1:
    path+='_test'

if not os.path.exists(path):
    os.mkdir(path)

path+='/hba_eryn_%d'%(args.id)

print(path)

if not os.path.exists(path):
    os.mkdir(path)

coords = {
    name: priors[name].rvs(size=(ntemps, nwalkers, nleaf,))
    for nleaf, name in zip(nleaves_max, branch_names)
}

meanchis=np.copy(np.asarray(coords['spins'][:,:,0,2]))
varchis=np.copy(np.asarray(coords['spins'][:,:,0,3]))

if args.sigma2==1:
    varchis=np.sqrt(varchis)

abetas=(meanchis-meanchis**2-varchis**2)*(meanchis/varchis**2)
bbetas=(meanchis-meanchis**2-varchis**2)*((1-meanchis)/varchis**2)

nfix_abeta=len(abetas[abetas<0])
nfix_bbeta=len(bbetas[bbetas<0])


while nfix_abeta>0 or nfix_bbeta>0:

    abetas=(meanchis-meanchis**2-varchis**2)*(meanchis/varchis**2)
    bbetas=(meanchis-meanchis**2-varchis**2)*((1-meanchis)/varchis**2)

    nfix_abeta=len(abetas[abetas<0])

    meanchis[abetas<0]=np.random.uniform(mins['meanchi'],maxs['meanchi'],nfix_abeta)
    varchis[abetas<0]=np.random.uniform(mins['varchi'],maxs['varchi'],nfix_abeta)

    if args.sigma2==1:
        varchis[abetas<0]=np.sqrt(varchis[abetas<0])

    nfix_bbeta=len(bbetas[bbetas<0])
    meanchis[bbetas<0]=np.random.uniform(mins['meanchi'],maxs['meanchi'],nfix_bbeta)
    varchis[bbetas<0]=np.random.uniform(mins['varchi'],maxs['varchi'],nfix_bbeta)

    if args.sigma2==1:
        varchis[bbetas<0]=np.sqrt(varchis[bbetas<0])

if args.sigma2==1:
    varchis=varchis**2

coords['spins'][:,:,0,2]=np.copy(meanchis)
coords['spins'][:,:,0,3]=np.copy(varchis)

inds = {
     name: np.random.randint(0,2,size=(ntemps, nwalkers, nleaf)).astype(bool)
     for nleaf, name in zip(nleaves_max, branch_names)
}

# if args.flat_leaves_gauss==1:
for branch in branch_names:
    for t in range(ntemps):
        for w in range(nwalkers):
            # ngauss_true=np.random.randint(ngauss_min,ngauss_max+1)
            inds[branch][t,w]=True
            # inds['gauss'][t,w,ngauss_true:]=False
            # np.random.shuffle(inds['gauss'][t,w])

state = State(coords, inds=inds)

factor = 0.01
if args.no_pl==0:
    cov = {"gauss": np.diag(np.ones(ndims[0])) * factor, "pl": np.diag(np.ones(ndims[1])) * factor, "global": np.diag(np.ones(ndims[2])) * factor,"spins": np.diag(np.ones(ndims[3])) * factor,\
    "q": np.diag(np.ones(ndims[4])) * factor, "z": np.diag(np.ones(ndims[5])) * factor}

# moves = GaussianMove(cov)
stretch_moves = StretchMove()
moves=[(stretch_moves,0.5)]

for branch in branch_names:
    stretch_move_i = StretchMove(gibbs_sampling_setup=branch)

    moves+=[(stretch_move_i,0.1)]

data = {key: xp.asarray(val) for key, val in data.items()}

ensemble = EnsembleSampler(nwalkers,
    ndims,  # assumes ndim_max
    loglike,
    priors,
    tempering_kwargs=dict(betas=betas),
    nbranches=len(branch_names),
    branch_names=branch_names,
    nleaves_max=nleaves_max,
    nleaves_min=nleaves_min,
    provide_groups=True,
    moves=moves,
    vectorize=True,
    rj_moves=True  # basic generation of new leaves from the prior
)

# breakpoint()

ensemble.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin_by)



samples_gauss = ensemble.get_chain()['gauss'][:,0].reshape(nsteps, nwalkers,dict_nleaves_max['gauss'],dict_dims['gauss'])


samples_pl = ensemble.get_chain()['pl'][:,0].reshape(nsteps, nwalkers, dict_nleaves_max['pl'],dict_dims['pl'])

samples_global = ensemble.get_chain()['global'][:,0].reshape(nsteps, nwalkers, dict_nleaves_max['global'],dict_dims['global'])

samples_spins = ensemble.get_chain()['spins'][:,0].reshape(nsteps, nwalkers, dict_nleaves_max['spins'],dict_dims['spins'])

samples_q = ensemble.get_chain()['q'][:,0].reshape(nsteps, nwalkers, dict_nleaves_max['q'],dict_dims['q'])

samples_z = ensemble.get_chain()['z'][:,0].reshape(nsteps, nwalkers, dict_nleaves_max['z'],dict_dims['z'])


samples_leaves_gauss=ensemble.get_nleaves()['gauss'][:, 0].reshape(nsteps,nwalkers,1)
samples_leaves_pl=ensemble.get_nleaves()['pl'][:, 0].reshape(nsteps,nwalkers,1)


npts=nsteps

thin=10
inds_keep=np.arange(npts)[::thin]

samples_gauss=samples_gauss[inds_keep]

samples_pl=samples_pl[inds_keep]
 

samples_global=samples_global[inds_keep]

samples_spins=samples_spins[inds_keep]

samples_q=samples_q[inds_keep]

samples_z=samples_z[inds_keep]

samples_leaves_gauss=samples_leaves_gauss[inds_keep]
samples_leaves_pl=samples_leaves_pl[inds_keep]

loglikes=ensemble.get_log_like()[:,0].reshape(nsteps,nwalkers,1)

loglikes=loglikes[::thin]

# breakpoint()

print(np.nanmax(loglikes))

np.savez(path+'/samples.npz',samples_gauss=samples_gauss,samples_leaves_gauss=samples_leaves_gauss,samples_pl=samples_pl,samples_leaves_pl=samples_leaves_pl,samples_global=samples_global,\
samples_spins=samples_spins,samples_q=samples_q,samples_z=samples_z,loglikes=loglikes)


# file_config.close()
