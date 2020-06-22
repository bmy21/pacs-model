import matplotlib
#the line below needs to be here so that matplotlib can save figures
#without an X server running - e.g. if using ssh/tmux
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogLocator, MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import numpy as np
from scipy.optimize import differential_evolution
from scipy.ndimage.interpolation import shift, rotate
from scipy.stats import anderson
from astropy.io import fits
from astropy.convolution import convolve_fft
from astropy.wcs import WCS
import pickle
import emcee
import corner
from rebin import congrid
from multiprocessing import Pool
from collections import namedtuple
import warnings
import tqdm
import os
import shutil


### Classes ###

#A simple data structure designed to hold upper limits on model parameters
ParamLimits = namedtuple('ParamLimits', ['fmax', 'shiftmax', 'rmax', 'imax'])


class Plottable:
    """Class representing an astronomical image (either real or synthetic), with plotting functionality."""

    def __init__(self, pfov, image, image_hires = None, hires_scale = np.nan):
        self.pfov = pfov                #pixel field of view in arcsec
        self.image = image
        self.image_hires = image_hires  #optional high-resolution version of the image
        self.hires_scale = hires_scale


    def __sub__(self, other):
        """Return a basic Plottable object whose image is the difference of self and other.
        Note the resulting object will not have an associated high-resolution image."""

        if not np.isclose(self.pfov, other.pfov, rtol = 1e-6):
            raise Exception("Tried to subtract two Plottables with different pixel sizes.")
        else:
            return Plottable(self.pfov, self.image - other.image)


    def _projected_sep_array(self, centre):
        """Make an array of projected separations from a particular point to each image pixel.

        Parameters
        ----------
        centre : 2D tuple of ints
            Indices of the point from which to calculate separations.

        Returns
        -------
        2D array
            Array of projected separations.
        """

        dx, dy = np.meshgrid(np.arange(self.image.shape[1]) - centre[1],
                             np.arange(self.image.shape[0]) - centre[0])

        return self.pfov * np.sqrt(dx**2 + dy**2)


    def consistent_gaussian(self, radius = None):
        """Establish whether self.image appears consistent with Gaussian noise.

        If a radius is provided, return True if either the whole image, or the region within
        radius arcsec of the centre, is consistent with Gaussian noise. The idea here is that
        checking only the central region reduces the likelihood of other sources influencing
        the result, but reducing the number of pixels also increases the significance of any
        bright pixels, so it's useful to check both regions."""

        if radius is not None:
            sky_separation = self._projected_sep_array([i/2 for i in self.image.shape])
            data = self.image[sky_separation < radius]

        else:
            data = self.image.flatten()

        #perform an Anderson-Darling test for normality
        result = anderson(data)

        #according to scipy documentation, [2] should correspond to the 5% level,
        #however this function is set up to account for possible future changes
        sig = result.significance_level[2]
        crit = result.critical_values[2]

        #return the significance level and whether the data are consistent with a Gaussian at that level
        if radius is None:
            return sig, (result.statistic < crit)
        else:
            return sig, (result.statistic < crit or self.consistent_gaussian(None)[1])


    def _find_brightest(self, sep_threshold, centre):
        """Find the brightest pixel within a specified angular distance of a given pixel in self.image.

        Parameters
        ----------
        sep_threshold : float
            Radius of search region in arcsec.
        centre : 2D tuple of ints
            Indices of centre of search region.

        Returns
        -------
        2D tuple of ints
            Indices of the brightest pixel.
        """

        sky_separation = self._projected_sep_array(centre)
        return np.unravel_index(np.ma.MaskedArray(self.image, sky_separation > sep_threshold).argmax(),
                                                  self.image.shape)


    def _get_limits(self):
        """Calculate the appropriate limits in arcsec for a plot of the object's image."""

        return [self.image.shape[1] * self.pfov / 2, -self.image.shape[1] * self.pfov / 2,
                -self.image.shape[0] * self.pfov / 2, self.image.shape[0] * self.pfov / 2]


    def plot(self, ax, plot_hires = False, xlabel = True, ylabel = False, log = False,
                   annotation = '', cmap = 'inferno', scalebar = False, dist = np.nan):
        """Plot the given image using the provided axes, and add a colorbar below."""

        #change units from mJy/pix to mJy/arcsec^2
        intensity_scale = ((self.hires_scale if plot_hires else 1)/ self.pfov)**2

        #need to make a copy since image will be manipulated if log == True
        image = self.image_hires.copy() if plot_hires else self.image.copy()

        #set zero pixels to arbitrary small amount (half the smallest pixel) to allow log plotting
        if log:
            if np.any(image < 0) or np.all(image == 0):
                warnings.warn("Cannot plot this image on a log scale. Using a linear scale.",
                              stacklevel = 2)
                log = False
            else:
                image[image == 0] = np.amin(image[image > 0]) / 2

        if xlabel: ax.set_xlabel('$\mathregular{RA\ offset\ /\ arcsec}$')
        if ylabel: ax.set_ylabel('$\mathregular{Dec\ offset\ /\ arcsec}$')

        ax.tick_params(direction = 'in', color = 'white', width = 1, right = True, top = True)

        limits = self._get_limits()

        im = ax.imshow(np.log10(image * intensity_scale) if log else image * intensity_scale,
                       origin = 'lower',
                       interpolation = 'none', cmap = cmap,
                       extent = limits)


        #put an annotation at the top left corner
        ax.annotate(annotation, xy = (0.05, 0.95), xycoords = 'axes fraction', color = 'white',
                    verticalalignment = 'top', horizontalalignment = 'left')

        #add a scalebar if desired
        if scalebar:
            if np.isnan(dist):
                warnings.warn("No distance provided to plot_image. Unable to plot a scale bar.",
                              stacklevel = 2)

            else:
                #place scalebar at the lower left corner
                scalebar_x = 0.05 #axis fraction
                scalebar_y = 0.05 #axis fraction

                #choose an appropriate scalebar length
                if dist < 5:
                    scalebar_au = 10
                elif dist < 1000:
                    scalebar_au = 100
                else:
                    scalebar_au = 1000

                scalebar_arcsec = scalebar_au / dist

                ax.plot([limits[0] + scalebar_x * (limits[1] - limits[0]),
                         limits[0] + scalebar_x * (limits[1] - limits[0]) - scalebar_arcsec],
                        [limits[2] + scalebar_y * (limits[3] - limits[2]) for i in range(2)],
                        color = 'white')

                ax.annotate(f'{scalebar_au} au',
                            xy = (scalebar_x + scalebar_arcsec / abs(limits[1] - limits[0]) + 0.02,
                                  scalebar_y),
                            xycoords = 'axes fraction', color = 'white',
                            verticalalignment = 'center', horizontalalignment = 'left')

        #add a colorbar
        if not log: cblabel = '$\mathregular{Intensity\ /\ (mJy\ arcsec^{-2})}$'
        else: cblabel = '$\mathregular{log\ [\ Intensity\ /\ (mJy\ arcsec^{-2})\ ]}$'

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size = '5%', pad = 0.6)
        cb = plt.colorbar(im, cax = cax, orientation = 'horizontal')
        cb.set_label(cblabel)

        cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda x , pos: f'{x:g}'))
        cb.ax.xaxis.set_minor_locator(plt.NullLocator())
        cb.ax.xaxis.set_major_locator(MaxNLocator(nbins = 5))

        #return the relevant AxesImage
        return im


    def plot_contours(self, ax, rms, levels = [-3, -2, 2, 3], neg_col = 'gainsboro', pos_col = 'k'):
        """Plot contours showing the specified RMS levels of the given image."""

        limits = self._get_limits()

        #return the relevant QuadContourSet
        return ax.contour(np.linspace(limits[0], limits[1], self.image.shape[1]),
                          np.linspace(limits[2], limits[3], self.image.shape[0]),
                          self.image, [i * rms for i in levels],
                          colors = [neg_col, neg_col, pos_col, pos_col], linestyles = 'solid')


class Model(Plottable):
    def __init__(self, params, shape, pfov, aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor):
        """Store the defining properties of the model."""

        (self.funres, self.fres, self.x0, self.y0,
         self.r1, self.r2, self.inc, self.theta) = params if include_unres else np.concatenate(([0], params))

        self.pfov = pfov
        self.aupp = aupp
        self.hires_scale = hires_scale
        self.alpha = alpha
        self.stellarflux = stellarflux
        self.flux_factor = flux_factor
        self.shape = shape


    def _distances_hires(self):
        """Calculate distance from the star to the disc element at each model pixel.

        Returns
        -------
        2D array
            Distances in au calculated on a grid of dimensions (shape * hires_scale).
        """

        #note the use of +self.x0 but -self.y0, since RA increases to the left
        dx, dy = np.meshgrid(np.linspace(-self.aupp * self.shape[1] / 2, self.aupp * self.shape[1] / 2,
                                         num = self.shape[1] * self.hires_scale) + self.x0,
                             np.linspace(-self.aupp * self.shape[0] / 2, self.aupp * self.shape[0] / 2,
                                         num = self.shape[0] * self.hires_scale) - self.y0)

        #'primed' coordinates, i.e. in a frame rotated by theta
        dxpr = dx * np.cos(np.deg2rad(self.theta)) + dy * np.sin(np.deg2rad(self.theta))
        dypr = -dx * np.sin(np.deg2rad(self.theta)) + dy * np.cos(np.deg2rad(self.theta))

        return np.sqrt(dypr**2 + (dxpr / np.cos(np.deg2rad(self.inc)))**2)


    def _make_hires(self):
        """Make a high-resolution image of the model.

        Returns
        -------
        2D array
            Model flux on a grid of dimensions (shape * hires_scale), in mJy/pixel.
        """

        #set disc flux based on d^(-alpha) profile
        d = self._distances_hires()
        in_disc = (d > self.r1) & (d < self.r2)
        flux = np.zeros(d.shape)
        flux[in_disc] = d[in_disc] ** (-self.alpha)

        #ensure we don't divide by zero (if there's no flux, we don't need to normalize anyway)
        if np.sum(flux) > 0:
            flux = self.fres * flux / np.sum(flux)

        #scale down for lost flux
        flux /= self.flux_factor

        #at this point, store the flux without the central bright pixel in self.image_hires
        #to improve contrast if plotted later
        self.image_hires = flux.copy()

        #central pixel gets additional flux from star plus any unresolved flux
        index_central = np.unravel_index(np.argmin(d), d.shape)
        flux[index_central] += (self.stellarflux + self.funres) / self.flux_factor

        #now return the flux including the central pixel for processing by make_images
        return flux


    def make_images(self, psf_hires):
        """Store appropriate images for analysis/plotting in self.image and self.image_hires.

        Parameters
        ----------
        psf_hires : 2D array
            Image to use as PSF for the convolved image, interpolated to the high-resolution
            model pixel size.
        """

        #convolve high-resolution model with high-resolution PSF; note that the call to
        #_make_hires stores self.image_hires
        convolved_hires = convolve_fft(self._make_hires(), psf_hires)

        #rebin to lower-resolution image pixel size
        self.image = self.hires_scale**2 * congrid(convolved_hires, self.shape)


class Observation(Plottable):
    """Class representing a PACS observation. Stores the image and some associated metadata."""

    def __init__(self, filename, search_radius = 5, target_ra = np.nan, target_dec = np.nan, dist = np.nan,
                 boxsize = 13, hires_scale = 1, rotate_to = np.nan, normalize = False):
        """Load in an image, store some important parameters and perform initial image processing."""

        with fits.open(filename) as fitsfile:
            self.image = fitsfile['image'].data * 1000              #image in mJy/pixel
            self.pfov = fitsfile['image'].header['CDELT2'] * 3600   #pixel FOV in arcsec
            self.wav = int(fitsfile['PRIMARY'].header['WAVELNTH'])  #wavelength of observations
            self.level = int(fitsfile['PRIMARY'].header['LEVEL'])   #processing level (20 or 25)
            self.name = fitsfile['PRIMARY'].header['OBJECT']        #target name
            self.angle = fitsfile['PRIMARY'].header['POSANGLE']     #pointing position angle

            #extract the obsid; the appropriate keyword seemingly depends on the processing level
            try:
                self.obsid = fitsfile['PRIMARY'].header['OBSID001'] #this works for level 2.5
            except KeyError:
                self.obsid = fitsfile['PRIMARY'].header['OBS_ID']   #and this for level 2

            #get the expected star coordinates in pixels, if RA and dec were provided;
            #otherwise, assume it's at the centre of the image
            if np.isnan(target_ra) or np.isnan(target_dec):
                star_expected = [i / 2 for i in self.image.shape]
            else:
                star_expected = np.flip(WCS(fitsfile['image'].header).wcs_world2pix([[target_ra, target_dec]], 0)[0])

            #extract coverage level, so that we can estimate the rms flux in a suitable region
            cov = fitsfile['coverage'].data

        #refuse to analyse 160 micron data (70/100 is always available and generally at higher S/N)
        if self.wav != 70 and self.wav != 100:
            raise Exception(f"Please provide a 70 or 100 μm image ({filename} is at {self.wav} μm).")

        #factors to correct for flux lost during high-pass filtering (see Kennedy et al. 2012)
        if self.wav == 70:
            self.flux_factor = 1.16
        elif self.wav == 100:
            self.flux_factor = 1.19

        #if no distance is supplied, simply set d = 1 pc so that separations will be in arcsec, not au;
        #in_au can be stored in any saved output for future reference, and plots can be annotated with sep_unit,
        #which is intended to be embedded in a LaTeX string
        if np.isnan(dist):
            dist = 1
            self.sep_unit = r'^{\prime\prime}'
            self.in_au = False
        else:
            self.sep_unit = r'\mathrm{au}'
            self.in_au = True

        #au per pixel at the distance of the target
        self.aupp = self.pfov * dist

        #clean up NaN pixels
        self.image[np.isnan(self.image)] = 0
        cov[np.isnan(cov)] = 0

        #find the coordinates of the brightest pixel within search_radius arcsec
        #of the specified RA and dec (or simply the centre)
        brightest_pix = self._find_brightest(search_radius, star_expected)

        #estimate the rms flux in a region defined by two conditions: coverage is above a
        #specified level, and projected separation from the brightest pixel is above a certain level.
        #NOTE: if the provided RA/dec are far from the image centre, the region defined by these
        #conditions may not be the most appropriate (however, it's unlikely that we will be trying
        #to fit a source near the edge of a map)

        cov_threshold_rms = 0.6 #fraction of max coverage
        sep_threshold_rms = 15 #arcsec
        sky_separation = self._projected_sep_array(brightest_pix)

        self.rms = self._estimate_background((cov > cov_threshold_rms * np.max(cov)) &
                                             (sky_separation > sep_threshold_rms))

        #need to scale up uncertainties since noise is correlated
        natural_pixsize = 3.2 #always the case for PACS 70/100 micron images
        self.uncert = self.rms * self._correlated_noise_factor(natural_pixsize)

        #cut out a portion of the image with the brightest pixel at the centre
        self._crop_image(brightest_pix, 2 * boxsize)

        #rotate to a particular position angle if requested (necessary if using image as a PSF)
        if not np.isnan(rotate_to): self.image = rotate(self.image, self.angle - rotate_to)

        #now cut down to the requested size; note that we again look for the brightest pixel
        #and put this in the centre, since the rotation may have introduced a small offset
        self._crop_image(self._find_brightest(2 * self.pfov, [i / 2 for i in self.image.shape]), boxsize)

        #normalize if requested
        if normalize: self.image /= np.sum(self.image)

        #rebin to a higher resolution if requested
        self.hires_scale = hires_scale
        if self.hires_scale > 1:
            self.image_hires = congrid(self.image,
                                      [i * self.hires_scale for i in self.image.shape],
                                      minusone = True)

            #ensure that flux is conserved
            self.image_hires *= np.sum(self.image) / np.sum(self.image_hires)

        elif hires_scale < 1:
            raise Exception(f'hires_scale should be an integer >= 1.')


    def _crop_image(self, centre, boxscale):
        """Crop self.image such that the specified pixel is perfectly centred.

        Parameters
        ----------
        centre : 2D tuple of ints
            Indices of pixel to place at the centre.
        boxscale : int
            Cut out a square of dimension (2 * boxscale + 1).
        """

        #this function guarantees that the desired pixel will be exactly at the centre
        #as long as boxscale < centre[i], which is useful for making centred plots

        #TODO: implement this in a simpler way?

        self.image = self.image[np.ix_(range(int(centre[0] - boxscale), int(centre[0] + boxscale + 1)),
                                range(int(centre[1] - boxscale), int(centre[1] + boxscale + 1)))]


    def _estimate_background(self, condition = None, sigma_level = 3.0, tol = 1e-6, max_iter = 20):
        """Estimate the background RMS of self.image or self.image[condition] using an iterative method.

        Parameters
        ----------
        condition : 2D array, optional
            Boolean array specifying which pixels of self.image to process. (default: None)
        sigma_level : float
            Number of standard deviations used to define outliers. (default: 3.0)
        tol : float, optional
            Fractional difference between RMS iterations at which to cease iterating. (default: 1e-6)
        max_iter : int, optional
            Maximum number of iterations. (default: 20)

        Returns
        -------
        float
            Estimated background RMS.
        """

        data = self.image.flatten() if condition is None else self.image[condition]
        rms = np.std(data)

        #arbitrarily set error to >tol so that the iterations can get started
        err = 2 * tol

        i = 0
        while (err > tol) and (i < max_iter):
            rmsprev = rms

            #discard data classified as outliers
            data = data[data < sigma_level * rms]
            rms = np.std(data)

            err = abs((rms - rmsprev) / rmsprev)
            i += 1

        if (i == max_iter) and (err > tol):
            warnings.warn(f"_estimate_background did not converge after {max_iter} iterations."
                          " You may wish to check the image for issues.", stacklevel = 2)

        return rms


    def _correlated_noise_factor(self, natural_pixsize):
        """Calculate the uncertainty scale factor for correlated noise from Fruchter & Hook (2002).

        Parameters
        ----------
        natural_pixsize : float
            Natural pixel size of the image in arcsec.

        Returns
        -------
        float
            Uncertainty scale factor.
        """

        r = natural_pixsize / self.pfov

        if r >= 1.0:
            return r / (1.0 - 1.0 / (3.0 * r))
        else:
            return 1.0 / (1.0 - r / 3.0)


### Functions used during initial setup ###

def choose_psf(level, wav):
    """Returns a path to one of four default PSFs based on processing level and wavelength."""

    #TODO: put this in a class/struct?

    if wav == 70:
        if level == 20:
            name_psf = ('psf/gamma_dra_70/1342217404/level2/HPPPMAPB/'
                        'hpacs1342217404_20hpppmapb_00_1469423089198.fits.gz')
        elif level == 25:
            name_psf = ('psf/gamma_dra_70/1342217404/level2_5/HPPHPFMAPB/'
                        'hpacs_25HPPHPFMAPB_blue_1757_p5129_00_v1.0_1470980845846.fits.gz')
        else:
            raise Exception(f'No level {level} PSF is provided by default.')

    elif wav == 100:
        if level == 20:
            name_psf = ('psf/gamma_dra_100/1342216069/level2/HPPPMAPB/'
                        'hpacs1342216069_20hpppmapb_00_1469417766626.fits.gz')
        elif level == 25:
            name_psf = ('psf/gamma_dra_100/1342216069/level2_5/HPPHPFMAPB/'
                        'hpacs_25HPPHPFMAPB_green_1757_p5129_00_v1.0_1470967312171.fits.gz')
        else:
            raise Exception(f'No level {level} PSF is provided by default.')

    else:
        raise Exception(f'No {wav} μm PSF is provided by default.')

    return name_psf


### Functions used for disc model fitting ###

#TODO: remove these two functions (need to change shifted psf)
def radius_limit(shape, aupp, phys = 2000):
    """Return a sensible limit on the disc's outer radius in au."""

    #make sure the disc doesn't lie entirely outside the image field of view;
    #also impose a physical limit in au, which is 2000 by default
    #(see e.g. Fig 3 of Hughes et al. 2018)
    return min(min(shape) * aupp / np.sqrt(2), phys)


def param_limits(shape, aupp):
    """Return limits defining the box in which to optimize parameters.
    The image shape must be supplied to make sure the disc doesn't lie
    outside the image."""

    rmax = radius_limit(shape, aupp) #in au

    #keep the disc's x/y offsets within +/- shiftmax pixels of the model origin
    shiftmax = 5 #in PACS pixels

    fmax = 30000 #i.e. 30Jy, to be safe

    #simple geometric model breaks down at 90 deg inclination, so limit to
    #some high value < 90 deg for now
    imax = 88

    return fmax, shiftmax, rmax, imax


def chi2(params, psf, alpha, include_unres, stellarflux, obs, param_limits):
    """Subtract model from image and calculate the chi-squared value, with uncertainties given by uncert."""

    model = Model(params, obs.image.shape, obs.pfov, obs.aupp, psf.hires_scale,
                  alpha, include_unres, stellarflux, obs.flux_factor)

    #impose uniform priors within some ranges;
    #note that the fluxes don't have an upper limit here, to allow for extremely bright cases
    if (model.funres < 0 or model.fres < 0
        or model.r1 <= 0 or model.r2 <= 0
        or model.r1 > param_limits.rmax or model.r2 > param_limits.rmax
        or model.inc < 0 or model.inc > param_limits.imax
        or abs(model.x0) > param_limits.shiftmax * obs.aupp
        or abs(model.y0) > param_limits.shiftmax * obs.aupp
        or abs(model.theta) > 90):
        return np.inf

    #force the disc to be at least a model pixel wide (otherwise completely unphysical models with
    #just a few pixels scattered around the image can result)
    dr_pix = (model.r2 - model.r1) * np.cos(np.deg2rad(model.inc)) * (psf.hires_scale / obs.aupp)
    if (model.r1 >= model.r2 or dr_pix <= 1):
        return np.inf

    model.make_images(psf.image_hires)
    return np.sum(((obs.image - model.image) / obs.uncert) ** 2)


def log_probability(params, *args):
    """Log-probability to be maximized by MCMC."""

    return -0.5 * chi2(params, *args)


def shifted_psf(psf, params):
    """Shift the provided PSF by offset (in pixels) and normalize its peak to scale."""

    x0, y0, scale = params
    model = shift(psf, [x0, y0])

    return model * scale / np.amax(model)


def chi2_shifted_psf(params, image, psf, uncert):
    """Calculate the chi-squared value associated with a PSF-subtracted image."""

    model = shifted_psf(psf, params)

    return np.sum(((image - model) / uncert) ** 2)


def best_psf_subtraction(image, psf, uncert):
    """Return the best-fitting PSF-subtracted image."""

    _, shiftmax, _, _ = param_limits(image.shape, 0)

    #note that shiftmax here is in PACS pixels
    limits = [(-shiftmax, shiftmax), (-shiftmax, shiftmax), (0, 2 * np.amax(image))]

    result = differential_evolution(chi2_shifted_psf, limits, args = (image, psf, uncert))

    return image - shifted_psf(psf, result['x'])



### Main functions ###

def save_params(savepath, resolved, include_unres = None, max_likelihood = None, median = None,
                lower_uncertainty = None, upper_uncertainty = None, model_consistent = None,
                in_au = None, stellarflux = None):
    """Save the main results of the fit in a pickle file."""

    dict = {
        "resolved": resolved,
        "include_unres": include_unres,
        "max_likelihood": max_likelihood,
        "median": median,
        "lower_uncertainty": lower_uncertainty,
        "upper_uncertainty": upper_uncertainty,
        "model_consistent": model_consistent,
        "in_au": in_au,
        "stellarflux": stellarflux
    }

    with open(savepath + '/params.pickle', 'wb') as file:
        pickle.dump(dict, file, protocol = pickle.HIGHEST_PROTOCOL)

    return


def run(name_image, name_psf = '', savepath = 'pacs_model/output/', name = '', dist = np.nan,
        stellarflux = 0, boxsize = 13, hires_scale = 5, alpha = 1.5, include_unres = False,
        initial_steps = 100, nwalkers = 200, nsteps = 800, burn = 600, ra = np.nan,
        dec = np.nan, test = False):
    """Fit one image and save the output."""

    #if given no stellar flux, force an unresolved component to be added
    if (stellarflux == 0 or np.isnan(stellarflux)) and not include_unres:
        include_unres = True
        warnings.warn("No stellar flux was supplied. Forcing the model to include an unresolved flux.",
                      stacklevel = 2)

    obs = Observation(name_image, target_ra = ra, target_dec = dec, dist = dist, boxsize = boxsize)

    #put the star name, obsid/level and wavelength together into an annotation for the image plot
    #TODO: ensure that a name always gets chosen (ie from FITS if necessary)
    annotation = '\n'.join([f'{obs.wav} μm image (level {(obs.level/10):g})', f'ObsID: {obs.obsid}', name])


    #if no PSF is provided, select one based on the processing level and wavelength
    if name_psf == '':
        name_psf = choose_psf(obs.level, obs.wav)

    psf = Observation(name_psf, boxsize = boxsize, hires_scale = hires_scale, rotate_to = obs.angle,
                      normalize = True)

    #abort execution if the PSF pixel scale doesn't match that of the image
    if not np.isclose(psf.pfov, obs.pfov, rtol = 1e-6):
        raise Exception("PSF and image pixel sizes do not match.")

    #issue a warning if the image and PSF are at different wavelengths
    if psf.wav != obs.wav:
        warnings.warn("The wavelength of the supplied PSF does not match that of the image.",
                      stacklevel = 2)

    #before starting to save output, remove any old files in the output folder
    if os.path.exists(savepath):
        shutil.rmtree(savepath)

    os.makedirs(savepath)


    #if requested, first check whether the image is consistent with a PSF and skip the fit if possible
    psfsub = Plottable(image = best_psf_subtraction(obs.image, psf.image, obs.uncert), pfov = obs.pfov)

    if test:
        sig, is_noise = psfsub.consistent_gaussian(15)

        if is_noise:
            print(f"The PSF subtraction is consistent with Gaussian noise at the {sig:.0f}% level."
                  " There is likely not a resolved disc here. Skipping this system.")

            print("Exporting image of PSF subtraction...")

            #make a two-panel image: [data, psf subtraction]
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 6), sharey = True)

            #first plot the PACS data
            obs.plot(ax[0], annotation = annotation)

            #then the PSF subtraction
            psfsub.plot(ax[1], ylabel = False, annotation = 'PSF subtraction')
            psfsub.plot_contours(ax[1], obs.rms)

            plt.tight_layout()
            fig.savefig(savepath + '/image_model.png', dpi = 150)
            #plt.show()
            plt.close(fig)

            #save a pickle simply indicating that no disc was resolved
            save_params(savepath, False)

            return

        else:
            print(f"The PSF subtraction is not consistent with Gaussian noise at the {sig:.0f}% level."
                  " There may be a resolved disc here. Performing disc fit.")


    #find best-fitting parameters using differential evolution, which searches for the
    #global minimum within the parameter ranges specified by the arguments.
    #format is [<funres,> fres, x0, y0, r1, r2, inc, theta]

    #note that the radius is restricted either to 2000 au (see e.g. Fig 3 of Hughes et al. 2018)
    #or to the half-diagonal length of the image - this is a quick way of ensuring that we don't
    #end up with arbitrarily large discs that lie completely outside the image cutout (which is
    #a problem because such discs can have arbitrarily large fluxes)
    param_limits = ParamLimits(
                               fmax = 200000,   #mJy (rare but there are a few systems this bright)
                               shiftmax = 5,    #PACS pixels
                               rmax = min(min(obs.image.shape) * obs.aupp / np.sqrt(2), 2000), #au
                               imax = 88        #deg
                              )

    search_space = [(0, param_limits.fmax), (0, param_limits.fmax),
                    (-param_limits.shiftmax * obs.aupp, param_limits.shiftmax * obs.aupp),
                    (-param_limits.shiftmax * obs.aupp, param_limits.shiftmax * obs.aupp),
                    (0, param_limits.rmax), (0, param_limits.rmax),
                    (0, param_limits.imax), (-90, 90)]

    pnames = [r'$F_\mathrm{unres}\ /\ \mathrm{mJy}$', r'$F_\mathrm{res}\ /\ \mathrm{mJy}$',
              fr'$x_0\ /\ {obs.sep_unit}$', fr'$y_0\ /\ {obs.sep_unit}$',
              fr'$r_1\ /\ {obs.sep_unit}$', fr'$r_2\ /\ {obs.sep_unit}$',
              r'$i\ /\ \mathrm{deg}$', r'$\theta\ /\ \mathrm{deg}$'] #parameter names for plot labels

    #if not including an unresolved flux, remove the first element of the parameter list
    if not include_unres:
        search_space.pop(0)
        pnames.pop(0)
        pnames[0] = r'$F_\mathrm{disc}\ /\ \mathrm{mJy}$'


    print("Finding a suitable initial model...")

    pbar = tqdm.tqdm(total = initial_steps)

    #set tol = 0 to ensure that DE runs for the prescribed number of steps & the progress bar works
    res = differential_evolution(chi2, search_space,
                                args = (psf, alpha, include_unres, stellarflux, obs, param_limits),
                                updating = 'deferred', workers = -1, #use multiprocessing
                                tol = 0, popsize = 20, maxiter = initial_steps, polish = False,
                                callback = (lambda xk, convergence: pbar.update()))

    pbar.close()

    p0 = res['x']
    ndim = p0.size

    print("Running MCMC sampler...")

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args = (psf, alpha, include_unres, stellarflux, obs, param_limits),
                                        pool = pool)

        #initialize the walkers with an ndim-dimensional Gaussian distribution
        pos = [p0 + p0 * 0.01 * np.random.randn(ndim) for i in range(nwalkers)]

        #run MCMC
        pos, prob, state = sampler.run_mcmc(pos, nsteps, progress = True)


    print("Pickling samples...")

    #extract and save the samples, excluding burn-in, for future use in e.g. corner plots
    samples = sampler.get_chain(discard = burn, flat = True)

    with open(savepath + '/samples.pickle','wb') as file:
        pickle.dump(samples, file, protocol = pickle.HIGHEST_PROTOCOL)


    print("Exporting plot of MCMC chains...")

    #save a plot of the chains
    fig, ax = plt.subplots(ndim + 1, figsize = (12, 16))
    chain = sampler.get_chain()

    #first plot the parameter chains
    for i in range(ndim):
        ax[i].plot(chain[:, :, i], c = 'k', alpha = 0.3)
        ax[i].set_ylabel(pnames[i])
        ax[i].xaxis.set_major_locator(plt.NullLocator())


    #then the log-probability
    ax[-1].plot(sampler.get_log_prob(), c = 'k', alpha = 0.3)
    ax[-1].xaxis.set_major_locator(MaxNLocator(integer = True))
    ax[-1].set_ylabel('log probability')
    ax[-1].set_xlabel('Step number')

    #formatting common to all subplots
    for i in range(ndim + 1):
        ax[i].axvspan(0, burn - 0.5, alpha = 0.1, color = 'k')
        ax[i].set_xlim(0, nsteps - 1)

    plt.tight_layout(h_pad = 0.5)
    fig.savefig(savepath + '/chains.pdf')
    plt.close(fig)


    print("Exporting corner plot...")

    #make the corner plot
    fig = corner.corner(samples, quantiles = [0.16, 0.50, 0.84],
                        labels = pnames, show_titles = True, title_fmt = '.1f')
    fig.savefig(savepath + '/corner.pdf')
    plt.close(fig)


    #get max-likelihood model parameters (no need to exclude burn-in for this, just want the best fit)
    max_likelihood = sampler.flatchain[np.argmax(sampler.flatlnprobability), :]

    #get median and 16th/84th percentile parameters
    median = np.median(samples, axis = 0)
    lower_uncertainty = median - np.percentile(samples, 16, axis = 0)
    upper_uncertainty = np.percentile(samples, 84, axis = 0) - median


    #now make a four-panel image: [data, psf subtraction, high-res max-likelihood model, residuals]
    print("Exporting image of best-fit model...")

    model = Model(max_likelihood, obs.image.shape, obs.pfov, obs.aupp, hires_scale, alpha,
                  include_unres, stellarflux, obs.flux_factor)

    model.make_images(psf.image_hires)

    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (18, 6), sharey = True)

    #first plot the PACS data
    obs.plot(ax[0], annotation = annotation)

    #then the PSF subtraction
    psfsub.plot(ax[1], ylabel = False, annotation = 'PSF subtraction')
    psfsub.plot_contours(ax[1], obs.rms)

    #now the high-res model
    annotation_model = 'High-resolution model'
    annotation_model += f'\nUnresolved component{" " if include_unres else " not "}included'

    if not obs.in_au:
        annotation_model += f'\nNo distance{" or stellar flux " if stellarflux == 0 else " "}provided'
    elif stellarflux == 0:
        annotation_model += '\nNo stellar flux provided'

    model.plot(ax[2], plot_hires = True, annotation = annotation_model, ylabel = False,
               log = True, scalebar = obs.in_au, dist = dist)


    #finally, the model residuals
    residual = obs - model
    residual.plot(ax[3], ylabel = False, annotation = 'Residuals')
    residual.plot_contours(ax[3], obs.rms)

    plt.tight_layout()
    fig.savefig(savepath + '/image_model.png', dpi = 150)
    plt.show()
    plt.close(fig)


    #check whether the model appears to be a good fit
    sig, is_noise = residual.consistent_gaussian(15)

    if is_noise:
        print(f"The residuals are consistent with Gaussian noise at the {sig:.0f}% significance level."
              " The disc model appears to explain the data well.")
    else:
        print(f"The residuals are not consistent with Gaussian noise at the {sig:.0f}% significance level."
              " You may wish to check the residuals for issues.")


    #finally, save the important parameters in a pickle for future analysis
    #note that stellarflux is saved so that we can check whether it was zero & hence how to interpret
    #the model fluxes (i.e. disc flux vs total system flux)
    save_params(savepath, True, include_unres, max_likelihood, median, lower_uncertainty, upper_uncertainty,
                is_noise, obs.in_au, stellarflux)


def parse_args():
    """Parse command-line arguments and return the results as a tuple."""

    parser = argparse.ArgumentParser(description = 'Fit a debris disc model to a Herschel PACS image.',
                                     formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', dest = 'img', metavar = 'image_file',
                        help = 'path to FITS file containing image to fit', required = True)
    parser.add_argument('-o', dest = 'output', metavar = 'output',
                        help = 'directory to place output (default ./pacs_model/output/)', default = 'pacs_model/output/')
    parser.add_argument('-n', dest = 'name', metavar = 'name',
                        help = 'name of the star, used as a figure annotation if supplied', default = '')
    parser.add_argument('-p', dest = 'psf', metavar = 'psf_file',
                        help = 'optional path to FITS file containing image to use as PSF', default = '')
    parser.add_argument('-d', dest = 'dist', type = float, metavar = 'distance',
                        help = 'distance in pc (if not provided, disc scale will be in \'\')', default = np.nan)
    parser.add_argument('-f', dest = 'fstar', type = float, metavar = 'stellar_flux',
                        help = 'stellar flux from synthetic photometry in mJy (default 0)', default = 0)
    parser.add_argument('-b', dest = 'boxsize', type = int, metavar = 'boxsize',
                        help = 'image cutout has dimension 2 * boxsize + 1 (default 13)', default = 13)
    parser.add_argument('-m', dest = 'model_scale', type = int, metavar = 'model_scale',
                        help = 'PACS pixel / high-res model pixel size ratio (default 5)', default = 5)
    parser.add_argument('-a', dest = 'alpha', type = float, metavar = 'alpha',
                        help = 'surface brightness profile index (d^-alpha; default 1.5)', default = 1.5)
    parser.add_argument('-s', dest = 'initial_steps', type = int, metavar = 'init_steps',
                        help = 'number of steps for initial optimization (default 100)', default = 100)
    parser.add_argument('-mw', dest = 'walkers', type = int, metavar = 'mcwalkers',
                        help = 'number of MCMC walkers (default 200)', default = 200)
    parser.add_argument('-ms', dest = 'steps', type = int, metavar = 'mcsteps',
                        help = 'number of MCMC steps (default 800)', default = 800)
    parser.add_argument('-mb', dest = 'burn', type = int, metavar = 'mcburn',
                        help = 'number of MCMC steps to discard as burn-in (default 600)', default = 600)
    parser.add_argument('-ra', dest = 'ra', type = float, metavar = 'ra',
                        help = 'target right ascension in degrees (optional)', default = np.nan)
    parser.add_argument('-de', dest = 'dec', type = float, metavar = 'dec',
                        help = 'target declination in degrees (optional)', default = np.nan)
    parser.add_argument('-u', dest = 'unres', action = 'store_true',
                        help = 'if set, include a component of unresolved flux in the model')
    parser.add_argument('-t', dest = 'testres', action = 'store_true',
                        help = 'if set, test whether the system appears consistent with a point source and skip disc fit if so')

    args = parser.parse_args()

    #MCMC parameters
    nwalkers = args.walkers
    nsteps = args.steps
    burn = args.burn

    #filenames
    name_image = args.img
    name_psf = args.psf

    #stellar parameters
    dist = args.dist
    stellarflux = args.fstar
    name = args.name
    ra = args.ra
    dec = args.dec

    #model parameters
    initial_steps = args.initial_steps
    alpha = args.alpha
    boxsize = args.boxsize
    include_unres = args.unres

    #make high-resolution model with (hires_scale*hires_scale) sub-pixels per PACS pixel
    hires_scale = args.model_scale

    #where to save the results
    savepath = args.output

    #should we use PSF subtraction to test for a resolved disc?
    test = args.testres

    return (name_image, name_psf, savepath, name, dist, stellarflux, boxsize, hires_scale, alpha, include_unres,
            initial_steps, nwalkers, nsteps, burn, ra, dec, test)


#allow command-line execution
if __name__ == "__main__":
    run(*parse_args())
