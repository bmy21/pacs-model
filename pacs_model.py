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
import warnings
import tqdm
import os
import shutil


### Classes ###

class Model:
    def __init__(self, params, shape, aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor):
        #set the Model's parameters based on the given list & whether an unresolved component is included
        (self.funres, self.fres, self.x0, self.y0,
         self.r1, self.r2, self.inc, self.theta) = params if include_unres else np.concatenate(([0], params))

        self.aupp = aupp
        self.hires_scale = hires_scale
        self.alpha = alpha
        self.stellarflux = stellarflux
        self.flux_factor = flux_factor
        self.shape = shape

        return


    def distances_hires(self):
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


    def make_hires(self, include_central = True):
        """Make a high-resolution image of the model.

        Parameters
        ----------
        include_central : bool, optional
            Should the model include the central bright pixel? This should be enabled
            when comparing the model with observations, but disabling it when plotting
            can give images with better contrast. (default: True)

        Returns
        -------
        2D array
            Model flux on a grid of dimensions (shape * hires_scale), in mJy/pixel.
        """

        #set disc flux based on d^(-alpha) profile
        d = self.distances_hires()
        in_disc = (d > self.r1) & (d < self.r2)
        flux = np.zeros(d.shape)
        flux[in_disc] = d[in_disc] ** (-self.alpha)

        #ensure we don't divide by zero (if there's no flux, we don't need to normalize anyway)
        if np.sum(flux) > 0:
            flux = self.fres * flux / np.sum(flux)

        #central pixel gets additional flux from star plus any unresolved flux
        if include_central:
            index_central = np.unravel_index(np.argmin(d), d.shape)
            flux[index_central] += self.stellarflux + self.funres

        return flux / self.flux_factor


    def synthetic_obs(self, psf_hires):
        """Make a synthetic observation of a disc.

        Parameters
        ----------
        psf_hires : 2D array
            Image to use as PSF, interpolated to the high-resolution model pixel size.

        Returns
        -------
        2D array
            Synthetic PACS image in mJy/pixel.
        """

        #convolve high-resolution model with high-resolution PSF
        convolved_hires = convolve_fft(self.make_hires(), psf_hires)

        #rebin to lower-resolution image pixel size
        return self.hires_scale**2 * congrid(convolved_hires, self.shape)



### Functions used during initial setup ###

def correlated_noise_factor(natural, pfov):
    """Calculate the uncertainty scale factor for correlated noise from Fruchter & Hook (2002).

    Parameters
    ----------
    natural : float
        Natural pixel scale of the image in arcsec.
    pfov : float
        Actual pixel scale of the image in arcsec.

    Returns
    -------
    float
        Uncertainty scale factor.
    """

    r = natural / pfov

    if r >= 1.0:
        return r / (1.0 - 1.0 / (3.0 * r))
    else:
        return 1.0 / (1.0 - r / 3.0)


def projected_sep_array(shape, centre, pfov):
    """Make an array of projected separations from a particular point.

    Parameters
    ----------
    shape : 2D tuple of ints
        Shape of the array to create.
    centre : 2D tuple of ints
        Indices of the point from which to calculate separations.
    pfov : float
        Pixel scale of the image in arcsec.

    Returns
    -------
    2D array
        Array of projected separations.
    """

    dx, dy = np.meshgrid(np.arange(shape[1]) - centre[1], np.arange(shape[0]) - centre[0])
    return pfov * np.sqrt(dx**2 + dy**2)


def find_brightest(image, sky_separation_threshold, pfov, centre = None):
    """Find the brightest pixel within a specified distance of some point in an image.

    Parameters
    ----------
    image : 2D array
        Image to search.
    sky_separation_threshold : float
        Radius of search region in arcsec.
    pfov : float
        Pixel scale of the image in arcsec.
    centre : 2D tuple of ints, optional
        Indices of centre of search region. If not provided, search around image centre.

    Returns
    -------
    2D tuple of ints
        Indices of the brightest pixel.
    """

    if centre is None: centre = [i/2 for i in image.shape]

    sky_separation = projected_sep_array(image.shape, centre, pfov)
    return np.unravel_index(np.ma.MaskedArray(image, sky_separation > sky_separation_threshold).argmax(),
                                              image.shape)


def crop_image(image, centre, boxscale):
    """Crop an image such that the specified pixel is perfectly centred.

    Parameters
    ----------
    image : 2D array
        Image to crop.
    centre : 2D tuple of ints
        Indices of pixel to place at the centre.
    boxscale : int
        Cut out a square of dimension (2 * boxscale + 1).

    Returns
    -------
    2D array
        Cropped image.
    """

    #this function guarantees that the desired pixel will be exactly at the centre
    #as long as boxscale < centre[i]; this is useful for making centred plots
    return image[np.ix_(range(int(centre[0] - boxscale), int(centre[0] + boxscale + 1)),
                        range(int(centre[1] - boxscale), int(centre[1] + boxscale + 1)))]


def estimate_background(data, sigma_level = 3.0, tol = 1e-6, max_iter = 10):
    """Estimate the background RMS of the given data using an iterative method.

    Parameters
    ----------
    data : array
        Data whose background RMS should be estimated.
    sigma_level : float
        Number of standard deviations used to define outliers. (default: 3.0)
    tol : float, optional
        Fractional difference between RMS iterations at which to cease iterating. (default: 1e-6)
    max_iter : int, optional
        Maximum number of iterations. (default: 10)

    Returns
    -------
    float
        Estimated background RMS.
    """

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
        warnings.warn(f"estimate_background did not converge after {max_iter} iterations."
                      " You may wish to check the image for issues.", stacklevel = 2)

    return rms


def choose_psf(level, wav):
    """Returns a path to one of four default PSFs based on processing level and wavelength."""

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


def chi2(params, psf_hires, aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor, image, uncert):
    """Subtract model from image and calculate the chi-squared value, with uncertainties given by uncert."""

    model = Model(params, image.shape, aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor)

    #TODO: change the rmax limitation to a restriction on zero flux models
    _, shiftmax, rmax, imax = param_limits(image.shape, aupp)

    #impose uniform priors within some ranges
    #(note that the fluxes don't have an upper limit here)
    if (model.funres < 0 or model.fres < 0
        or model.r1 <= 0 or model.r2 <= 0 or model.r1 > rmax or model.r2 > rmax
        or model.inc < 0 or model.inc > imax
        or abs(model.x0) > shiftmax * aupp or abs(model.y0) > shiftmax * aupp
        or abs(model.theta) > 90):
        return np.inf

    #force the disc to be at least a model pixel wide
    dr_pix = (model.r2 - model.r1) * np.cos(np.deg2rad(model.inc)) * (hires_scale / aupp)
    if (model.r1 >= model.r2 or dr_pix <= 1):
        return np.inf

    return np.sum(((image - model.synthetic_obs(psf_hires)) / uncert) ** 2)


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


def consistent_gaussian(image, radius = None, pfov = None):
    """Establish whether the given image is consistent with a Gaussian distribution.

    If a radius is provided, return True if either the whole image, or the region within
    radius arcsec of the centre, is consistent with Gaussian noise. The idea here is that
    checking only the central region reduces the likelihood of other sources influencing
    the result, but reducing the number of pixels also increases the significance of any
    bright pixels, so it's useful to check both regions."""

    if radius is not None:
        if pfov is None:
            warnings.warn("consistent_gaussian received radius but no pfov; assuming pfov = 1.",
                          stacklevel = 2)
            pfov = 1

        sky_separation = projected_sep_array(image.shape, [i/2 for i in image.shape], pfov)
        data = image[sky_separation < radius]

    else:
        data = image.flatten()

    #perform an Anderson-Darling test for normality
    result = anderson(data)

    #according to scipy documentation, this should correspond to the 5% level;
    #however, this function is set up to account for possible future changes
    sig = result.significance_level[2]
    crit = result.critical_values[2]

    #return the significance level and whether the data are consistent with a Gaussian at that level
    if radius is None:
        return sig, (result.statistic < crit)
    else:
        return sig, (result.statistic < crit or consistent_gaussian(image, None, None)[1])


### Functions used for plotting ###

def standard_form(x, pos):
    """Put x into a standard-form string; can be used for formatting axis labels."""

    #extract the mantissa and exponent
    a, b = (f'{x:.2e}'.split('e'))
    a = float(a)
    b = int(b)

    if a == 0:
        #otherwise 0 will always appear as 0x10^0, which can look strange
        return '$0$'
    else:
        return fr'${a:g} \times 10^{{ {b:g} }}$'


def get_limits(image, scale, pfov):
    """Calculate the appropriate limits in arcsec for a plot of the given image."""

    return [(image.shape[1] / scale) * pfov / 2, -(image.shape[1] / scale) * pfov / 2,
            -(image.shape[0] / scale) * pfov / 2, (image.shape[0] / scale) * pfov / 2]


def plot_image(ax, image, pfov, scale = 1, xlabel = True, ylabel = False, log = False,
               annotation = '', cmap = 'inferno', scalebar = False, scalebar_au = 100, dist = None):
    """Plot the given image using the provided axes, and add a colorbar below."""

    #change units from mJy/pix to mJy/arcsec^2
    intensity_scale = (scale / pfov)**2

    if xlabel: ax.set_xlabel('$\mathregular{RA\ offset\ /\ arcsec}$')
    if ylabel: ax.set_ylabel('$\mathregular{Dec\ offset\ /\ arcsec}$')

    ax.tick_params(direction = 'in', color = 'white', width = 1, right = True, top = True)

    limits = get_limits(image, scale, pfov)

    im = ax.imshow(np.log10(image * intensity_scale) if log else image * intensity_scale,
                   origin = 'lower',
                   interpolation = 'none', cmap = cmap,
                   extent = limits)


    #put an annotation at the top left corner
    ax.annotate(annotation, xy = (0.05, 0.95), xycoords = 'axes fraction', color = 'white',
                verticalalignment = 'top', horizontalalignment = 'left')

    #add a scalebar if desired
    #TODO: automatically decide on an appropriate length for the scalebar?
    if scalebar:
        if dist is None:
            warnings.warn("No distance provided to plot_image. Unable to plot a scale bar.",
                          stacklevel = 2)

        elif dist == 1:
            #safe to assume that if dist = 1 is passed then the distance scale is
            #arcsec, since there are no real stars at <= 1 pc
            warnings.warn("Skipping au scale bar as plot_image received dist = 1.",
                          stacklevel = 2)

        else:
            #place scalebar at the lower left corner
            scalebar_x = 0.05 #axis fraction
            scalebar_y = 0.05 #axis fraction

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


def plot_contours(ax, image, pfov, rms, scale = 1, levels = [-3, -2, 2, 3], neg_col = 'gainsboro', pos_col = 'k'):
    """Plot contours showing the specified RMS levels of the given image."""

    limits = get_limits(image, scale, pfov)

    #return the relevant QuadContourSet
    return ax.contour(np.linspace(limits[0], limits[1], image.shape[1]),
                      np.linspace(limits[2], limits[3], image.shape[0]),
                      image, [i * rms for i in levels],
                      colors = [neg_col, neg_col, pos_col, pos_col], linestyles = 'solid')


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

    return (name_image, name_psf, savepath, name, dist, stellarflux, hires_scale, alpha, include_unres,
            initial_steps, nwalkers, nsteps, burn, ra, dec, test)


def run(name_image, name_psf = '', savepath = 'pacs_model/output/', name = '', dist = np.nan,
        stellarflux = 0, boxsize = 13, hires_scale = 5, alpha = 1.5, include_unres = False,
        initial_steps = 100, nwalkers = 200, nsteps = 800, burn = 600, ra = np.nan,
        dec = np.nan, test = False):
    """Fit one image and save the output."""

    #if given no stellar flux, force an unresolved component to be added
    if stellarflux == 0 and not include_unres:
        include_unres = True
        warnings.warn("No stellar flux was supplied; including an unresolved flux in the model.",
                      stacklevel = 2)

    #if no distance is supplied, simply set d = 1 pc so that r1, r2, x0 and y0 will be in arcsec, not au;
    #in_au will be stored in the saved output for future reference, and plots are annotated with sep_unit,
    #which is intended to be embedded in a LaTeX string
    if np.isnan(dist):
        dist = 1
        sep_unit = r'^{\prime\prime}'
        in_au = False
    else:
        sep_unit = r'\mathrm{au}'
        in_au = True

    #load in image file and extract some important data
    with fits.open(name_image) as image_datafile:
        #extract image from FITS file
        image_data = image_datafile[1].data * 1000 #convert to mJy/pixel

        pfov = image_datafile[1].header['CDELT2'] * 3600 #pixel FOV in arcsec

        wav = int(image_datafile[0].header['WAVELNTH']) #wavelength of observations

        #processing level (20 or 25)
        level = int(image_datafile[0].header['LEVEL'])

        #extract the obsid; the appropriate keyword depends on the processing level
        try:
            obsid = image_datafile[0].header['OBSID001'] #this works for level 2.5
        except KeyError:
            obsid = image_datafile[0].header['OBS_ID'] #and this for level 2

        #if a star name wasn't provided, use the target name from the observation header
        if name == '':
            name = image_datafile[0].header['OBJECT']

        #position angle of pointing
        image_angle = image_datafile[0].header['POSANGLE']

        #get the expected star coordinates in pixels, if RA and dec were provided
        if not np.isnan(ra) and not np.isnan(dec):
            star_indices = np.flip(WCS(image_datafile[1].header).wcs_world2pix([[ra, dec]], 0)[0])
        else:
            star_indices = None

        #extract coverage level, so that we can estimate the rms flux using a suitable region of the sky
        cov = image_datafile['coverage'].data

    #remove NaN pixels
    image_data[np.isnan(image_data)] = 0
    cov[np.isnan(cov)] = 0

    #au per pixel
    aupp = pfov * dist

    #factors to correct for flux lost during high-pass filtering (see Kennedy et al. 2012)
    if wav == 70:
        flux_factor = 1.16
    elif wav == 100:
        flux_factor = 1.19
    else:
        #refuse to fit 160 micron data (70/100 is always available and generally at higher S/N)
        raise Exception("Please provide a 70 or 100 micron image.")

    #put the star name, obsid/level and wavelength together into an annotation for the image plot
    annotation = '\n'.join([f'{wav} μm image (level {(level/10):g})', f'ObsID: {obsid}', name])

    #scale up uncertainties since noise is correlated
    natural_pixsize = 3.2 #for PACS 70/100 micron images
    uncert_scale = correlated_noise_factor(natural_pixsize, pfov)

    #in portion of image with low coverage (i.e. towards edges), there are high levels of noise
    cov_lower_bound = 0.6 * np.max(cov)

    #find the coordinates of the star, assuming it's at the brightest pixel within
    #star_search_radius arcsec of the image centre (or the specified RA and dec)
    star_search_radius = 12
    reference_indices = find_brightest(image_data, star_search_radius, pfov, star_indices)

    #find stellocentric distance to each pixel, so that we can cut out the
    #star (& disc) for the purposes of calculating the rms flux
    rms_sep_threshold = 20
    sky_separation = projected_sep_array(image_data.shape, reference_indices, pfov)

    #estimate the background rms flux
    rms = estimate_background(image_data[(cov > cov_lower_bound) & (sky_separation > rms_sep_threshold)])

    #only interested in the star (& disc) at the centre of the image,
    #so cut out that part to save time
    image_data = crop_image(image_data, reference_indices, boxsize)

    #if no PSF is provided, select one based on the processing level and wavelength
    if name_psf == '':
        name_psf = choose_psf(level, wav)


    #load in the PSF
    with fits.open(name_psf) as psf_datafile:
        psf_data = psf_datafile[1].data * 1000 #convert to mJy/pixel
        psf_pfov = psf_datafile[1].header['CDELT2'] * 3600
        psf_wav = int(psf_datafile[0].header['WAVELNTH'])
        psf_angle =  psf_datafile[0].header['POSANGLE']

    psf_data[np.isnan(psf_data)] = 0

    #abort execution if the PSF pixel scale doesn't match that of the image
    if not np.isclose(psf_pfov, pfov, rtol = 1e-6):
        raise Exception("PSF and image pixel sizes do not match.")

    #issue a warning if the image and PSF are at different wavelengths
    if wav != psf_wav:
        warnings.warn("The wavelength of the supplied PSF does not match that of the image.",
                      stacklevel = 2)



    #cut out the PSF, rotate it to the appropriate orientation, and normalize it
    psf_boxsize = boxsize
    angle = psf_angle - image_angle
    psf_data = rotate(psf_data, angle)
    psf_data = crop_image(psf_data, find_brightest(psf_data, star_search_radius, pfov), psf_boxsize)
    psf_data /= np.sum(psf_data)

    #rebin PSF to pixel scale of high-resolution model, then re-normalize
    psf_data_hires = congrid(psf_data, [i * hires_scale for i in psf_data.shape], minusone = True)
    psf_data_hires /= np.sum(psf_data_hires)

    #before starting to save output, remove any old files in the output folder
    if os.path.exists(savepath):
        shutil.rmtree(savepath)

    os.makedirs(savepath)


    #if requested, first check whether the image is consistent with a PSF and skip the fit if possible
    psfsub = best_psf_subtraction(image_data, psf_data, rms * uncert_scale)

    if test:
        #test the central part of the image for normality - seems most sensible to take the radius as
        #rms_sep_threshold since we assumed everything outside that part is noise earlier. if we test
        #the entire image, the result will be affected by any background sources.
        sig, is_noise = consistent_gaussian(psfsub, rms_sep_threshold, pfov)

        if is_noise:
            print(f"The PSF subtraction is consistent with Gaussian noise at the {sig:.0f}% level."
                  " There is likely not a resolved disc here. Skipping this system.")

            print("Exporting image of PSF subtraction...")

            #make a two-panel image: [data, psf subtraction]
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9, 6), sharey = True)

            plot_image(ax[0], image_data, pfov, ylabel = True, annotation = annotation)
            plot_image(ax[1], psfsub, pfov, annotation = 'PSF subtraction')

            plot_contours(ax[1], psfsub, pfov, rms)

            plt.tight_layout()
            fig.savefig(savepath + '/image_model.png', dpi = 150)
            #plt.show()
            plt.close(fig)

            #for consistency, save a pickle indicating that no disc was resolved
            save_params(savepath, False)

            return

        else:
            print(f"The PSF subtraction is not consistent with Gaussian noise at the {sig:.0f}% level."
                  " There may be a resolved disc here. Performing disc fit.")


    #find best-fitting parameters using differential evolution, which searches for the
    #global minimum within the parameter ranges specified by the arguments.
    #format is [<funres,> fres, x0, y0, r1, r2, inc, theta]

    fmax, shiftmax, rmax, imax = param_limits(image_data.shape, aupp)
    shiftmax *= aupp
    search_space = [(0, fmax), (0, fmax), (-shiftmax, shiftmax), (-shiftmax, shiftmax),
                    (0, rmax), (0, rmax), (0, imax), (-90, 90)]

    pnames = [r'$F_\mathrm{unres}\ /\ \mathrm{mJy}$', r'$F_\mathrm{res}\ /\ \mathrm{mJy}$',
              fr'$x_0\ /\ {sep_unit}$', fr'$y_0\ /\ {sep_unit}$',
              fr'$r_1\ /\ {sep_unit}$', fr'$r_2\ /\ {sep_unit}$',
              r'$i\ /\ \mathrm{deg}$', r'$\theta\ /\ \mathrm{deg}$'] #parameter names for plot labels

    #if we're not including an unresolved flux parameter, remove the first element
    #of the parameter list
    if not include_unres:
        search_space.pop(0)
        pnames.pop(0)
        pnames[0] = r'$F_\mathrm{disc}\ /\ \mathrm{mJy}$'

    popsize = 20 #population size for differential evolution
    maxiter = initial_steps #number of differential evolution iterations

    print("Finding a suitable initial model...")

    pbar = tqdm.tqdm(total = maxiter)

    #set tol = 0 to ensure that DE runs for the prescribed number of steps & the progress bar works
    res = differential_evolution(chi2, search_space,
                                (psf_data_hires,
                                aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor,
                                image_data, rms * uncert_scale),
                                updating = 'deferred', workers = -1,
                                tol = 0, popsize = popsize, maxiter = maxiter, polish = False,
                                callback = (lambda xk, convergence: pbar.update()))

    pbar.close()

    p0 = res['x']
    ndim = p0.size

    print("Running MCMC sampler...")

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args = (psf_data_hires,
                                        aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor,
                                        image_data, rms * uncert_scale),
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

    #plt.subplots_adjust(hspace = 0.05)
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

    model = Model(max_likelihood, image_data.shape, aupp, hires_scale, alpha,
                  include_unres, stellarflux, flux_factor)


    residual = image_data - model.synthetic_obs(psf_data_hires)

    #don't include the central bright pixel for display purposes
    hires_model = model.make_hires(include_central = False)


    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (18, 6), sharey = True)

    #first plot the PACS data
    plot_image(ax[0], image_data, pfov, ylabel = True, annotation = annotation)

    #then the PSF subtraction
    plot_image(ax[1], psfsub, pfov, annotation = 'PSF subtraction')
    plot_contours(ax[1], psfsub, pfov, rms)

    #now the high-res model; set zero pixels to small amount (half the smallest pixel) as plotting on log scale,
    #apart from in the special case where there's no resolved flux at all
    nonzero_flux = np.sum(hires_model) > 0

    if nonzero_flux:
        hires_model[hires_model <= 0] = np.amin(hires_model[hires_model > 0]) / 2

    annotation_model = 'High-resolution model'
    annotation_model += f'\nUnresolved component{" " if include_unres else " not "}included'

    if not in_au:
        annotation_model += f'\nNo distance{" or stellar flux " if stellarflux == 0 else " "}provided'
    elif stellarflux == 0:
        annotation_model += '\nNo stellar flux provided'

    plot_image(ax[2], hires_model, pfov, scale = hires_scale, annotation = annotation_model,
               log = nonzero_flux, scalebar = True, scalebar_au = 100 if dist < 1000 else 1000, dist = dist)

    #finally, the model residuals
    plot_image(ax[3], residual, pfov, annotation = 'Residuals')
    plot_contours(ax[3], residual, pfov, rms)

    plt.tight_layout()
    fig.savefig(savepath + '/image_model.png', dpi = 150)
    plt.show()
    plt.close(fig)


    #check whether the model appears to be a good fit
    sig, is_noise = consistent_gaussian(residual, rms_sep_threshold, pfov)

    if is_noise:
        print(f"The residuals are consistent with Gaussian noise at the {sig:.0f}% significance level."
              " The disc model appears to explain the data well.")
    else:
        print(f"The residuals are not consistent with Gaussian noise at the {sig:.0f}% significance level."
              " You may wish to check the residuals for issues.")


    #finally, save the important parameters in a pickle for future analysis
    #note that stellarflux is saved, so that we can check whether it was zero & hence how to interpret
    #the model fluxes (i.e. disc flux or total system flux?)
    save_params(savepath, True, include_unres, max_likelihood, median, lower_uncertainty, upper_uncertainty,
                is_noise, in_au, stellarflux)


if __name__ == "__main__":
    run(*parse_args())
