import json
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde as kde
from scipy.special import gamma, hyp2f1
from glob import glob
import os
import sys

print("Importing effective spin priors")
#sys.path.append('/home/tristan-bruel/Documents/Science/o4a-astro/GW241011_GW241110_data_release/notebooks/effective-spin-priors')
from priors import chi_effective_prior_from_isotropic_spins, chi_effective_prior_from_aligned_spins

#if os.path.exists('./effective-spin-priors/'):
#    print("Importing effective spin priors")
#    sys.path.append("./effective-spin-priors/")
#    from priors import chi_effective_prior_from_isotropic_spins, chi_effective_prior_from_aligned_spins

#else:
#    print("Repository `effective-spin-priors` not found, skipping import. Some figures may not be created")

# -*- coding: utf-8 -*-
#
#       Copyright 2020
#       Zoheyr Doctor <zoheyr.doctor@ligo.org>
#       Leo Tsukada <leo.tsukada@ligo.org>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

# ----------------- HARD CODED QUANTITIES FOR PLOTS ---------------------------

gw241011_color = '#1f78b4'
gw241110_color = '#33a02c'

class Bounded_2d_kde(kde):

    """
    Represents a two-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain with reflective boundary conditions
    """

    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None, *args, **kwargs):

        """
        Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param xlow: The lower x domain boundary.

        :param xhigh: The upper x domain boundary.

        :param ylow: The lower y domain boundary.

        :param yhigh: The upper y domain boundary.
        """

        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'

        super(Bounded_2d_kde, self).__init__(pts.T, *args, **kwargs)

        self._xlow = xlow
        self._xhigh = xhigh
        self._ylow = ylow
        self._yhigh = yhigh

    @property
    def xlow(self):
        """The lower bound of the x domain."""
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain."""
        return self._xhigh

    @property
    def ylow(self):
        """The lower bound of the y domain."""
        return self._ylow

    @property
    def yhigh(self):
        """The upper bound of the y domain."""
        return self._yhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'

        x, y = pts.T
        pdf = super(Bounded_2d_kde, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, y])

        if self.xhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, y])

        if self.ylow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2*self.ylow - y])

        if self.yhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2*self.yhigh - y])

        if self.xlow is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, 2*self.ylow - y])

            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, 2*self.yhigh - y])

        if self.xhigh is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, 2*self.ylow - y])
            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, 2*self.yhigh - y])

        return pdf

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[pts[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[pts[:, 1] > self.yhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results


def get_contour_heights(kde, x_data, y_data, dx=(0.1, 0.1), dy=(0.1, 0.1), nbins=(100, 99), probs=(0.90, 0.50)):

    """
    Helper function to get contour levels corresponding to 50% and 90% credible 2D posterior bounds

    Parameters
    ----------
    kde : `func`
        KDE function to evaluate probabilities over a regular grid
    x_data : `np.ndarray`
        x-axis values over which KDE was generated
    y_data : `np.ndarray`
        y-axis values over which KDE was generated
    dx : `tuple`
        Specifies how much padding to adopt on the left and right of the minimum and maximum values
        of `x_data`, respectively, when defining grid over which to KDE
    dy : `tuple`
        Specifies how much padding to adopt below and above the minimum and maximum values
        of `y_data`, respectively, when defining grid over which to KDE
    nbins : tuple
        Number of points in x- and y-directions at which to evaluate KDE when generating and plotting contours
    probs : tuple
        Tuple of central posterior probabilities at which to draw identify contour levels. Default (0.90, 0.5)

    Returns
    -------
    x_bins : `np.ndarray`
        x-axis grid points over which KDE was evaluated
    y_bins : `np.ndarray`
        y-axis grid points over which KDE was evaluated
    heights : `np.ndarray`
        2D probabilities over `x_bins` and `y_bins`
    levels : `tuple`
        Tuple of 90% and 50% credible levels, for use in contour plotting
    """

    # Define grid containing provided data, with desired padding
    x_bins = np.linspace(np.min(x_data) - dx[0], np.max(x_data) + dx[1], nbins[0])
    y_bins = np.linspace(np.min(y_data) - dy[0], np.max(y_data) + dy[1], nbins[1])
    X, Y = np.meshgrid(x_bins, y_bins)

    # Evaluate KDE over grid
    heights = kde(np.array([X.reshape(-1), Y.reshape(-1)]).T)

    # Sort grid points by decreasing probabilities, and integrate outward to obtain contour levels
    heights_1d = np.sort(heights.reshape(-1))[::-1]
    cs = [np.interp(p, np.cumsum(heights_1d)/np.sum(heights_1d), heights_1d) for p in probs]
    
    return x_bins, y_bins, heights, cs


def plot_1d_hist(ax, datasets, x_lim, n_bins,
    colors, xlabel=None, ylabel=None):

    """
    Helper function to produce diagonal, 1D marginal histograms in corner plots.
    This is specifically intended to be used when overplotting results from two events,
    like GW241011 and GW241110

    Parameters
    ----------
    datasets : list 
        List of arrays containing sets of posterior samples to be overplotted 
    x_lim : tuple
        Tuple containing min and max x-axis values to display
    n_bins : int
        Number of histogram bins
    colors : list
        Colors with which to plot datasets
    xlabel : str
        If provided, x-axis label
    ylabel : str
        If provided, y-axis label

    Returns
    -------
    None
    """

    hist_lw=1.5

    # Histogram datasets
    for i in range(len(datasets)):

        ax.hist(datasets[i],
            density=True,
            bins=np.linspace(*x_lim, n_bins),
            histtype='step',
            color=colors[i],
            lw=hist_lw)

    # Set limits
    ax.set_xlim(*x_lim)

    # Label if provided
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticklabels([])

    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])


def plot_2d_hist(ax, datasets, x_lim, y_lim, colors,
                 xlow = None, xhigh=None, ylow=None, yhigh=None,
                 xlabel=None, ylabel=None,
                 delta_x = (0.05, 0.05), delta_y = (0.05, 0.05),
                 nbins=(100, 99)):

    """
    Helper function to produce 2d subplots in corner plots.
    This is specifically intended to be used when overplotting results from two events,
    like GW241011 and GW241110

    Parameters
    ----------
    datasets : list 
        List of tuples, containing x- and y-data to be plotted. 
        E.g. [(dataset1_x, dataset1_y), (dataset2_x, dataset2_y), ...]
    x_lim : tuple
        Tuple containing min and max x-axis values to display
    y_lim : tuple
        Tuple containing min and max y-axis values to display
    colors : list
        Colors with which to plot datasets
    xlow : float or None
        If provided, the min x-value at which KDE will adopt reflective boundary conditions
    xhigh : float or None
        If provided, the max x-value at which KDE will adopt reflective boundary conditions
    ylow : float or None
        If provided, the min y-value at which KDE will adopt reflective boundary conditions
    yhigh : float or None
        If provided, the max y-value at which KDE will adopt reflective boundary conditions
    xlabel : str
        If provided, x-axis label
    ylabel : str
        If provided, y-axis label
    delta_x : tuple
        Distance beyond the minimum and maximum x-values of provided data at which KDE will be evaluated when drawing contours
    delta_y : tuple
        Distance beyond the minimum and maximum y-values of provided data at which KDE will be evaluated when drawing contours
    nbins : tuple
        Number of points in x- and y-directions at which to evaluate KDE when generating and plotting contours

    Returns
    -------
    None
    """

    # Loop across datasets
    zorders = range(len(datasets))
    for i, dataset_pair in enumerate(datasets):
    
        # For each dataset, extract x and y values
        x_samples = np.array(dataset_pair[0])
        y_samples = np.array(dataset_pair[1])

        # Construct KDE with desired boundary conditions
        kde_x_y = Bounded_2d_kde(
            np.array([x_samples, y_samples]).T,
            xlow=xlow,
            xhigh=xhigh,
            ylow=ylow,
            yhigh=yhigh
            )
    
        # Use KDE to evaluade 2D probabilities over a grid containing data, and the level defining the 90% credible contour
        x_bins, y_bins, heights, cs  = get_contour_heights(kde_x_y, x_samples, y_samples, delta_x, delta_y, nbins)

        # Plot contour
        ax.contour(x_bins, y_bins, heights.reshape(y_bins.size, x_bins.size),
                   levels=cs, colors=colors[i],
                   linewidths=(1, 1.5),
                   zorder = zorders[i])
        
    # Set limits
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    # Label
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticklabels([])

    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])

def p_q(q, q_min):

    """
    Helper function to evaluate the marginal prior on mass ratio q,
    given uniform priors on m1 and m2 above some minimum mass ratio q_min.

    Parameters
    ----------
    q : float or np.ndarray
        Mass ratio
    q_min : float
        Minimum mass ratio considered
    
    Returns
    -------
    p_q : float or np.ndarray
        Normalized probability densities
    """

    num = (1+q)**(2./5.) * q_min**(1./5.) * gamma(-2./5.)
    denom = q**(6./5.) * (
        3 * q_min**(1./5.) * gamma(-6./5.) * gamma(4./5.)
        + 5 * gamma(-2./5.) * hyp2f1(-2./5., -1./5., 4./5, -q_min)
    )
    return num/denom

def p_s1z(s1z, aMax):

    """
    Helper function to evaluate the marginal prior on the aligned spin component,
    given uniform-in-magnitude and isotropic-in-direction priors

    Parameters
    ----------
    s1z : float or np.ndarray
        Aligned spin value
    aMax : float
        Maximum allowed spin magnitude

    Returns
    -------
    p_s1z : float or np.ndarray
        Normalized probability densities
    """

    return np.log(aMax/np.abs(s1z))/(2*aMax)

def p_s1p(s1p, aMax):

    """
    Helper function to evaluate the marginal prior on the in-plane spin component,
    given uniform-in-magnitude and isotropic-in-direction priors

    Parameters
    ----------
    s1p : float or np.ndarray
        In-plane spin component 
    aMax : float
        Maximum allowed spin magnitude

    Returns
    -------
    p_s1p : float or np.ndarray
        Normalized probability densities
    """

    return np.arctan(np.sqrt(aMax**2-s1p**2)/s1p)/aMax

def p_Xeff(Xeffs, q_min, aMax):

    """
    Helper function to evaluate the marginal prior on the effective inspiral spin,
    given uniform-in-magnitude and isotropic-in-direction priors and uniform-in-component
    mass priors, subject to a cut in mass ratio

    Parameters
    ----------
    Xeffs : float or np.ndarray
        Effective inspiral spins 
    q_min : float
        Minimum mass ratio allowed in mass prior
    aMax : float
        Maximum allowed spin magnitude

    Returns
    -------
    p_Xeff : float or np.ndarray
        Normalized probability densities
    """

    q_grid = np.linspace(q_min, 1, 100)
    X, Q = np.meshgrid(Xeffs, q_grid)

    p_Xeff_q = np.ones((q_grid.size, Xeffs.size))
    p_Xeff_q *= p_q(Q, q_min)

    for i, q in enumerate(q_grid):
        p_Xeff_q[i, :] *= chi_effective_prior_from_isotropic_spins(q, aMax, Xeffs)

    p_Xeff = np.trapezoid(p_Xeff_q, q_grid, axis=0)
    return p_Xeff

def p_Xeff_alignedSpin(Xeffs, q_min, aMax):

    """
    Helper function to evaluate the marginal prior on the effective inspiral spin,
    given uniform-in-aligned spin priors and uniform-in-component
    mass priors, subject to a cut in mass ratio

    Parameters
    ----------
    Xeffs : float or np.ndarray
        Effective inspiral spins 
    q_min : float
        Minimum mass ratio allowed in mass prior
    aMax : float
        Maximum allowed spin magnitude

    Returns
    -------
    p_Xeff : float or np.ndarray
        Normalized probability densities
    """

    q_grid = np.linspace(q_min, 1, 100)
    X, Q = np.meshgrid(Xeffs, q_grid)

    p_Xeff_q = np.ones((q_grid.size, Xeffs.size))
    p_Xeff_q *= p_q(Q, q_min)

    for i, q in enumerate(q_grid):
        p_Xeff_q[i, :] *= chi_effective_prior_from_aligned_spins(q, aMax, Xeffs)

    p_Xeff = np.trapezoid(p_Xeff_q, q_grid, axis=0)
    return p_Xeff


def chi_lower_bound_vs_chi_effective(chi_eff):

    """
    Function defining minimum allowed primary spin magnitude as a function of chi-effective

    Parameters
    ----------
    chi_eff : array
        Array of chi_effective values

    Returns
    -------
    min_chi: array
        Minimum possible primary spin magnitude
    """

    min_chi = np.maximum(np.maximum(2*chi_eff-1, np.zeros_like(chi_eff)), -2*chi_eff-1)
    return min_chi


def s1z_bounds_vs_chi_effective(chi_eff):

    """
    Function defining minimum and maximum allowed aligned spin components as a function of chi-effective

    Parameters
    ----------
    chi_eff : array
        Array of chi_effective values

    Returns
    -------
    min_s1z: array
        Minimum possible primary aligned spin magnitudes
    max_s1z: array
        Maximum possible primary aligned spin magnitudes
    """

    min_s1z = np.maximum(2*chi_eff-1, -1*np.ones_like(chi_eff))
    max_s1z = np.minimum(2*chi_eff+1, np.ones_like(chi_eff))

    return min_s1z, max_s1z


def s1p_upper_bound_vs_chi_effective(chi_eff):

    """
    Function defining maximum allowed in-plane spin components as a function of chi-effective

    Parameters
    ----------
    chi_eff : array
        Array of chi_effective values

    Returns
    -------
    max_s1p: array
        Maximum possible primary in-plane spin magnitudes
    """

    max_s1p = np.where(chi_eff<-0.5, np.sqrt(1-(1+2*chi_eff)**2), 1)
    max_s1p = np.where(chi_eff>0.5, np.sqrt(1-(1-2*chi_eff)**2), max_s1p)

    return max_s1p


def s1z_bounds_vs_spin_magnitude(spin_mag):

    """
    Function defining minimum and maximum allowed aligned spin components as a function of primary spin magnitude 

    Parameters
    ----------
    spin_mag : array
        Array of spin magnitude values

    Returns
    -------
    min_s1z: array
        Minimum possible primary aligned spin magnitudes
    max_s1z: array
        Maximum possible primary aligned spin magnitudes
    """

    min_s1z = -spin_mag
    max_s1z = spin_mag

    return min_s1z, max_s1z


def s1p_upper_bound_vs_spin_magnitude(spin_mag):

    """
    Function defining maximum allowed in-plane spin components as a function of primary spin magnitude 

    Parameters
    ----------
    spin_mag : array
        Array of spin magnitude values

    Returns
    -------
    max_s1p: array
        Maximum possible primary in-plane spin magnitudes
    """

    max_s1p = spin_mag
    return max_s1p


def s1p_upper_bound_vs_s1z(s1z):

    """
    Function defining maximum allowed in-plane spin components as a function of primary aligned spin

    Parameters
    ----------
    s1z : array
        Array of aligned spin values

    Returns
    -------
    max_s1p: array
        Maximum possible primary in-plane spin magnitudes
    """

    max_s1p = np.sqrt(1-s1z**2)
    return max_s1p
