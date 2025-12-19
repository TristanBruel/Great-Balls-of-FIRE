import numpy as np
import math

# various functions that model stellar mass loss in FIRE

Zsun = 0.02

# adapted from gizmo, FIRE-2 version
def Z_for_stellar_evol(Z):
    return np.minimum(np.maximum(Z/0.02, 0.01), 3.)

# adapted from gizmo, FIRE-2 revised version (used for MassiveFIRE2 runs)
# mass returned per Gyr per mass SSP [ units Gyr^-1 ]
# see stellar_evolution.c:mechanical_fb_calculate_eventrates_Winds
# mass-loss/unit time converted to mass loss as follows:
# p *= All.StellarMassLoss_Rate_Renormalization * (dt*UNIT_TIME_IN_GYR);
# Note: one difference between MassiveFIRE2 and FIREbox version is that
# if(star_age<=0.001){p=4.76317*ZZ;} (MassiveFIRE2)
# f (star_age<=0.001){p=4.76317;} (FIREbox)
#
# this is identical to the older version used by FIREbox (I checked & tested)
# see mechanical_fb.c:determine_where_SNe_occur
# p *= All.GasReturnFraction * (
#      dt*0.001*All.UnitTime_in_Megayears/All.HubbleParam);
def wind_mass_loss_rate(age, Z):
    aa = np.atleast_1d(age)/1e9
    p = np.zeros_like(aa)
    ZZ = Z_for_stellar_evol(Z)

    sel = aa <= 0.001
    if np.sum(sel) > 0:
       p[sel] = 4.76317

    sel = (aa > 0.001) & (aa <= 0.0035) 
    if np.sum(sel) > 0:
        try:
            l_lgZZ = np.log10(ZZ[sel])
            l_ZZ = ZZ[sel]
        except IndexError:
            l_lgZZ = np.log10(ZZ)
            l_ZZ = ZZ
        p[sel] = 4.76317*l_ZZ*10**(
            1.838*(0.79+l_lgZZ)*(np.log10(aa[sel])-(-3.00)))

    sel = (aa > 0.0035) & (aa <= 0.1)
    if np.sum(sel) > 0:
        p[sel] = 29.4*(aa[sel]/0.0035)**(-3.25)+0.0041987

    sel = (aa > 0.1)
    if np.sum(sel) > 0:
        p[sel] = 0.41987*aa[sel]**(-1.1)/(12.9-np.log(aa[sel]))

    if len(p) == 1:
        return p[0] * 1e-9  # to return mass loss per year (not Gyr)
    return p * 1e-9


# analytical time integral of wind_mass_loss_rate
from scipy.special import expi
def wind_integral(age, Z):

    scalar=False
    if np.isscalar(age) or age.ndim == 0:
        scalar=True

    age = np.atleast_1d(age.astype(np.float64)/1e9)
    # conv to solar units + some caps
    Z = np.atleast_1d(Z_for_stellar_evol(Z.astype(np.float64)))

    def f_ML_0(age, Z): # if age <= 0.001
        ML = 4.76317 * age
        return ML 

    def f_ML_1(age, Z):  # if 0.001 < age <= 0.0035
        ML = f_ML_0(0.001, Z) 
        b = 1.838*(0.79+np.log10(Z))
        ML += 4.76317*Z * 10**(3*b) * (age**(1+b) - 0.001**(1+b)) / (1+b)
        return ML 

    def f_ML_2(age, Z):  # if 0.0035 < age <= 0.1
        ML = f_ML_1(0.0035, Z)
        ML += 29.4*0.0035*((age/0.0035)**(-2.25)-1)/(-2.25)
        ML += 0.0041987*(age-0.0035)
        return ML

    def f_ML_3(age, Z):  # if age > 0.1
        ML = f_ML_2(0.1, Z) 
        a = 0.41987
        b = -1.1
        c = 12.9
        ML += a * np.exp((1+b)*c) * (
            expi((1+b)*(np.log(0.1)-c)) - expi((1+b)*(np.log(age)-c)))
        return ML

    ML = np.zeros_like(age)

    sel = age <= 0.001
    if np.sum(sel) > 0:
        ML[sel] = f_ML_0(age[sel], Z[sel])

    sel = (age > 0.001) & (age <= 0.0035)
    if np.sum(sel) > 0:
        ML[sel] = f_ML_1(age[sel], Z[sel])

    sel = (age > 0.0035) & (age <= 0.1)
    if np.sum(sel) > 0:
        ML[sel] = f_ML_2(age[sel], Z[sel])

    sel = age > 0.1
    if np.sum(sel) > 0:
        ML[sel] = f_ML_3(age[sel], Z[sel])

    if scalar:
        ML = ML[0]

    return ML

# see stellar_evolution.c:mechanical_fb_calculate_eventrates_SNe
#     stellar_evolution.c:particle2in_addFB_SNe
def SN_mass_loss_rate(age):

    aa = np.atleast_1d(age)/1e9
    RSNe = np.zeros_like(aa)

    # some ages in Gyr used below
    agemin=0.003401
    agebrk=0.01037
    agemax=0.03753

    # calculate: NSNe/Myr *if* each SNe had exactly 10^51 ergs; really from
    # the energy curve [do this so we are guaranteed to get the correct SNe
    # energy] //
    # core-collapse rate [super-simple 2-piece constant]
    sel = (aa > agemin) & (aa <=agebrk)
    RSNe[sel] = 5.408e-4
    sel = (aa > agebrk) & (aa <=agemax)
    RSNe[sel] = 2.516e-4

    # Ia (prompt Gaussian+delay, Manucci+06)
    sel = (aa > agemax)
    RSNe[sel] = 5.3e-8 + 1.6e-5*np.exp(
        -0.5*((aa[sel]-0.05)/0.01)*((aa[sel]-0.05)/0.01))

    MSNe = 10.5 * np.ones_like(RSNe)  # average ejecta mass for single event
    sel = (aa > agemax)  # assume early SNe are core-collapse, later are Ia
    MSNe[sel] = 1.4

    RSNe *= 1e-6  # convert SN rate from per Myr to per year

    #if aa[0]>1:
    #    import pdb; pdb.set_trace()

    p = RSNe * MSNe  # return mass loss per year per mass SSP
    if len(p) == 1:
        return p[0]
    return p

# analytical time integral of SN_mass_loss_rate
from scipy.special import erf
def SN_integral(age):
    """age in yr"""

    scalar=False
    if np.isscalar(age) or age.ndim == 0:
        scalar=True

    age = np.atleast_1d(age.astype(np.float64))/1e9

    # some ages in Gyr used below
    agemin=0.003401
    agebrk=0.01037
    agemax=0.03753
    M_II = 10.5
    M_Ia = 1.4

    ML = np.zeros_like(age)

    sel = (age >= agemin) & (age < agebrk)
    if np.sum(sel) > 0:
        ML[sel] =  M_II * 5.408e-4*(age[sel]-agemin)

    sel = (age >= agebrk) & (age < agemax)
    if np.sum(sel) > 0:
        ML[sel] = (M_II * (5.408e-4*(agebrk-agemin) +
                   2.516e-4*(age[sel]-agebrk)))

    sel = age >= agemax
    if np.sum(sel) > 0:
        aux =  M_II * (5.408e-4*(agebrk-agemin) + 2.516e-4*(agemax-agebrk))
        s = 0.01
        s_times_sqrt2 = s * np.sqrt(2)
        age_max_tilde = (agemax - 0.05)/s_times_sqrt2
        pre_fac = 1.6e-5 * s_times_sqrt2 * np.sqrt(np.pi)

        age_tilde = (age[sel] - 0.05)/s_times_sqrt2
        ML[sel] = aux + M_Ia * (5.3e-8 * (age[sel]-agemax) +
                      pre_fac * 0.5 * (erf(age_tilde)-erf(age_max_tilde)))

    ML *= 1e3  # conversions between Gyr, Myr, and yr

    if scalar:
        ML = ML[0]

    return ML # integrated mass loss

def r_mass_loss_rate(age):
    aa = np.atleast_1d(age)/1e9
    p = np.zeros_like(aa)
    age_min = 0.003

    sel = aa >= age_min
    p[sel] = 3e-5 / (1000 * aa[sel]) # Nsne / Myr for a 1 Msun population 
    p *= 1e3  # Nsne / Gyr
    p *= 0.01  # 0.01 solar mass of r process elements 
    return p * 1e-9  # [ in yr^-1 ]

# analytical time integral of r_mass_loss_rate
# r_mass_loss rate is negligble compared with winds and SN
def r_integral(age):
    """age in yr"""

    scalar=False
    if np.isscalar(age) or age.ndim == 0:
        scalar=True

    age = np.atleast_1d(age.astype(np.float64))/1e9

    age_min = 0.003

    ML = np.zeros_like(age)
    sel = age >= age_min
    if np.sum(sel) > 0:
        ML[sel] = 0.01 * 1e3 * 3e-5 / 1000 * np.log(age[sel]/age_min)

    if scalar:
        ML = ML[0]

    return ML

# this is theoretically the correct way as dt => 0
def mass_loss(age, Z, fudge=1.0):
    """input: age in yr, Z in absolute units"""
    p_wind = wind_integral(age, Z)
    p_SN = SN_integral(age)
    p_r = r_integral(age)
    # int_t0^t1 p dt = - ln M(t1)/M(t0)
    ML = 1 - fudge*np.exp(-(p_wind+p_SN+p_r))
    return ML

def wind_mass_loss(age, Z, fudge=1.0):
    # p is d M / dt / M
    p_wind = wind_integral(age, Z)
    ML = 1 - fudge*np.exp(-p_wind)  # int_t0^t1 p dt =  ln M(t0)/M(t1)
    return ML

def SN_mass_loss(age, fudge=1.0):
    p_SN = SN_integral(age)
    ML = 1 - fudge*np.exp(-p_SN)
    return ML

def r_mass_loss(age, fudge=1.0):
    p_r = r_integral(age)
    ML = 1 - fudge*np.exp(-p_r)
    return ML

def mass_ratio(age, Z, fudge=1.0):
    """Return m(t=0)/m(t=age), i.e., multiply it by m(t=age) for orig. mass"""
    ML = mass_loss(age, Z, fudge=fudge)
    return 1/(1. - ML)
