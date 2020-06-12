import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogLocator, MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import differential_evolution
from scipy.ndimage.interpolation import shift, rotate
from scipy.stats import anderson
from astropy.io import fits
from astropy.convolution import convolve_fft
import pickle
import emcee
import corner
from rebin import congrid
from multiprocessing import Pool
import warnings
import tqdm
import os
import shutil


#a few PSF files we might want to try out later...
#name_psf='calibstar/1342182988/level2/HPPPMAPB/hpacs1342182988_20hpppmapb_00_1469207572390.fits.gz'
# psf1
#name_psf='calibstar/1342217404/level2_5/HPPHPFMAPB/hpacs_25HPPHPFMAPB_blue_1757_p5129_00_v1.0_1470980845846.fits.gz'
# psf2
#name_psf='calibstar/1342191959/level2_5/HPPHPFMAPB/hpacs_25HPPHPFMAPB_blue_1757_p5129_00_v1.0_1470609127545.fits.gz''


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


def find_brightest(image, sky_separation_threshold, pfov):
    """Find the brightest pixel within a specified distance of the centre of an image.

    Parameters
    ----------
    image : 2D array
        Image to search.
    sky_separation_threshold : float
        Radius of search region in arcsec.
    pfov : float
        Pixel scale of the image in arcsec.

    Returns
    -------
    2D tuple of ints
        Indices of the brightest pixel.
    """

    sky_separation = projected_sep_array(image.shape, [i/2 for i in image.shape], pfov)
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


### Functions used for disc model fitting ###

def fix_model_params(params, include_unres):
    """If using a model with no unresolved flux, prepend a zero to the parameter array.

    Parameters
    ----------
    params: ndarray
        Model parameters.
    include_unres : bool
        Should the model include an unresolved component?

    Returns
    -------
    ndarray
        Model parameters, with an additional zero prepended if necessary.
    """

    return params if include_unres else np.concatenate(([0], params))


def distance_array(params, shape, aupp, hires_scale, include_unres):
    """Calculate distance from the star to the disc element at each model pixel.

    Parameters
    ----------
    params : ndarray
        Model parameters.
    shape : 2D tuple of ints
        Shape of the PACS image to model.
    aupp : float
        Astronomical units covered by one PACS pixel.
    hires_scale : int
        Factor by which the model pixels are smaller than the PACS pixels.
    include_unres : bool
        Should the model include an unresolved component?

    Returns
    -------
    2D array
        Distances in au calculated on a grid of dimensions (shape * hires_scale).
    """


    funres, ftot, x0, y0, r1, r2, inc, theta = fix_model_params(params, include_unres)

    dtr = np.pi / 180.0

    dx, dy = np.meshgrid(np.linspace(-aupp * shape[1] / 2, aupp * shape[1] / 2,
                                     num = shape[1] * hires_scale) + x0, #+ as RA increases to left
                         np.linspace(-aupp * shape[0] / 2, aupp * shape[0] / 2,
                                     num = shape[0] * hires_scale) - y0)

    dxpr = dx * np.cos(theta * dtr) + dy * np.sin(theta * dtr)
    dypr = -dx * np.sin(theta * dtr) + dy * np.cos(theta * dtr)
    d = np.sqrt((dypr)**2 + ((dxpr) / np.cos(inc * dtr))**2)

    return d


def model_hires(params, shape, aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor, include_central = True):
    """Make a model image of a disc, with coordinates relative to the image centre.

    Parameters
    ----------
    params : ndarray
        Model parameters.
    shape : 2D tuple of ints
        Shape of the PACS image to model.
    aupp : float
        Astronomical units covered by one PACS pixel.
    hires_scale : int
        Factor by which the model pixels are smaller than the PACS pixels.
    alpha : float
        Power law index defining disc brightness profile.
    include_unres : bool
        Should the model include an unresolved component?
    stellarflux : float
        Expected stellar flux in the observation band, in mJy.
    flux_factor : float
        Factor by which to scale down the disc flux due to high-pass filtering.
    include_central : bool, optional
        Should the model include the central bright pixel? (default: True)

    Returns
    -------
    2D array
        Model flux on a grid of dimensions (shape * hires_scale), in mJy/pixel.
    """

    funres, ftot, x0, y0, r1, r2, inc, theta = fix_model_params(params, include_unres)

    #set disc flux based on d^(-alpha) profile
    d = distance_array(params, shape, aupp, hires_scale, include_unres)
    in_disc = (d > r1) & (d < r2)
    flux = np.zeros(d.shape)
    flux[in_disc] = d[in_disc] ** (-alpha)

    #ensure we don't divide by zero (if there's no flux, we don't need to normalize anyway)
    if np.sum(flux) > 0:
        flux = ftot * flux / np.sum(flux)

    #central pixel gets additional flux from star plus an optional component of unresolved flux
    if include_central:
        index_central = np.unravel_index(np.argmin(d), d.shape)

        flux[index_central] += stellarflux + funres

    return flux / flux_factor


def synthetic_obs(params, psf_hires, shape, aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor):
    """Make a synthetic observation of a disc.

    Parameters
    ----------
    params : ndarray
        Model parameters.
    psf_hires : 2D array
        Image to use as PSF, interpolated to the model pixel size.
    shape : 2D tuple of ints
        Shape of the PACS image to model.
    aupp : float
        Astronomical units covered by one PACS pixel.
    hires_scale : int
        Factor by which the model pixels are smaller than the PACS pixels.
    alpha : float
        Power law index defining disc brightness profile.
    include_unres : bool
        Should the model include an unresolved component?
    stellarflux : float
        Expected stellar flux in the observation band, in mJy.
    flux_factor : float
        Factor by which to scale down the disc flux due to high-pass filtering.

    Returns
    -------
    2D array
        Synthetic PACS image, with the same shape as the actual image, in mJy/pixel.
    """

    #convolve high-resolution model with high-resolution PSF
    model = convolve_fft(model_hires(params, shape, aupp, hires_scale,
                                     alpha, include_unres, stellarflux, flux_factor),
                         psf_hires)

    #rebin to lower-resolution image pixel size
    model = hires_scale**2 * congrid(model, shape, method = 'linear', centre = False, minusone = False)

    return model


def chi2(params, psf_hires, aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor, image, uncert):
    """Subtract model from image and calculate the chi-squared value, with uncertainties given by uncert."""

    funres, ftot, x0, y0, r1, r2, inc, theta = fix_model_params(params, include_unres)

    #force the disc to be at least a model pixel wide
    dr_pix = (r2 - r1) * np.cos(np.deg2rad(inc)) * (hires_scale / aupp)

    #disallow unphysical parameters
    if (funres < 0 or ftot < 0 or r1 >= r2 or dr_pix <= 1 or r1 <= 0 or r2 <= 0
        or inc < 0 or inc > 85 or abs(theta) > 90 or r2 > 1000):
        return np.inf

    model = synthetic_obs(params, psf_hires, image.shape, aupp,
                          hires_scale, alpha, include_unres, stellarflux, flux_factor)

    return np.sum(((image - model) / uncert)**2)


def log_probability(params, *args):
    """Log-probability to be maximized by MCMC."""

    return -0.5 * chi2(params, *args)


def shifted_psf(psf, offset, scale):
    """Shift the provided PSF by offset and normalize its peak to scale."""

    model = shift(psf, offset)
    return model * scale / np.amax(model)


def chi2_shifted_psf(offset, image, psf, uncert):
    """Calculate the chi-squared value associated with a PSF-subtracted image."""

    #only parameter to minimize over is the PSF offset
    #always scale the PSF so that its peak intensity matches that of the image
    model = shifted_psf(psf, offset, np.amax(image))

    return np.sum(((image - model) / uncert)**2)


def best_psf_subtraction(image, psf, uncert, pix = 5):
    """Return the best-fitting PSF-subtracted image. The PSF may be offset by +/- pix pixels."""

    result = differential_evolution(chi2_shifted_psf, [(-pix, pix), (-pix, pix)],
                                    args = (image, psf, uncert))

    return image - shifted_psf(psf, result['x'], np.amax(image))


def consistent_gaussian(data):
    """Establish whether the given data are consistent with a Gaussian distribution."""

    #perform an Anderson-Darling test for normality
    result = anderson(data)

    #according to scipy documentation, this should correspond to the 5% level;
    #however, this function is set up to account for possible future changes
    sig = result.significance_level[2]
    crit = result.critical_values[2]

    #return the sigficance level and whether the data are consistent with
    #a Gaussian at that level
    return sig, (result.statistic < crit)


### Functions used for plotting ###

def standard_form(x, pos):
    """Put x into a standard-form string; used for formatting colorbar labels."""

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

    norm = colors.LogNorm(vmin = np.amin(image) * intensity_scale,
                          vmax = np.amax(image) * intensity_scale) if log else None

    limits = get_limits(image, scale, pfov)

    im = ax.imshow(image * intensity_scale, origin = 'lower',
                   interpolation = 'none', cmap = cmap, norm = norm,
                   extent = limits)

    #put an annotation at the top left corner
    ax.annotate(annotation, xy = (0.05, 0.95), xycoords = 'axes fraction', color = 'white',
                verticalalignment = 'top', horizontalalignment = 'left')

    #add a scalebar if desired
    if scalebar:
        if dist is None:
            warnings.warn("No distance provided to plot_image. Unable to plot a scale bar.",
                          stacklevel = 2)

        else:
            #by default put the scalebar at the top left
            scalebar_x = 0.05 #axis fraction
            scalebar_y = 0.95 #axis fraction

            scalebar_arcsec = scalebar_au / (dist * pfov)

            #if an annotation was provided, plot at the bottom left instead
            if annotation != '':
                scalebar_y = 0.05

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
    cblabel = '$\mathregular{Intensity\ /\ (mJy\ arcsec^{-2})}$'

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size = '5%', pad = 0.6)
    cb = plt.colorbar(im, cax = cax, orientation = 'horizontal')
    cb.set_label(cblabel)

    cb.ax.xaxis.set_major_formatter(FuncFormatter(standard_form))
    cb.ax.xaxis.set_minor_locator(plt.NullLocator())
    if log: cb.ax.xaxis.set_major_locator(LogLocator(base = 2))

    #rotate labels to avoid overlap
    plt.setp(cb.ax.xaxis.get_majorticklabels(), rotation = 25)

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
                lower_uncertainty = None, upper_uncertainty = None, model_consistent = None):
    """Save the main results of the fit in a pickle."""

    with open(savepath + '/params.pickle', 'wb') as file:

        dict = {
            "resolved": resolved,
            "include_unres": include_unres,
            "max_likelihood": max_likelihood,
            "median": median,
            "lower_uncertainty": lower_uncertainty,
            "upper_uncertainty": upper_uncertainty,
            "model_consistent": model_consistent
        }

        pickle.dump(dict, file, protocol = pickle.HIGHEST_PROTOCOL)

    return

def parse_args():
    """Parse command-line arguments and return the results."""

    parser = argparse.ArgumentParser(description = 'Fit a model to a resolved PACS image of a debris disc.',
                                     formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-o', dest = 'output', metavar = 'output',
                        help = 'directory in which to save the output', default = 'pacs_model/output/')
    parser.add_argument('-n', dest = 'name', metavar = 'name',
                        help = 'name of the star, used to annotate the output', default = '')
    parser.add_argument('-i', dest = 'img', metavar = 'image_file',
                        help = 'path to FITS file containing image to fit', required = True)
    parser.add_argument('-p', dest = 'psf', metavar = 'psf_file',
                        help = 'path to FITS file containing image to use as PSF',
                        default = 'calibstar/1342217404/level2_5/HPPHPFMAPB/hpacs_25HPPHPFMAPB_blue_1757_p5129_00_v1.0_1470980845846.fits.gz')
    parser.add_argument('-d', dest = 'dist', type = float, metavar = 'distance',
                        help = 'distance to star in pc', required = True)
    parser.add_argument('-f', dest = 'fstar', type = float, metavar = 'stellar_flux',
                        help = 'stellar flux from synthetic photometry in mJy', required = True)
    parser.add_argument('-m', dest = 'model_scale', type = int, metavar = 'model_scale',
                        help = 'scale of high-resolution model relative to PACS image', required = True)
    parser.add_argument('-a', dest = 'alpha', type = float, metavar = 'alpha',
                        help = 'power-law index of surface brightness profile (d^-alpha)',
                        default = 1.5)
    parser.add_argument('-w', dest = 'walkers', type = int, metavar = 'walkers',
                        help = 'number of MCMC walkers', default = 200)
    parser.add_argument('-s', dest = 'steps', type = int, metavar = 'steps',
                        help = 'number of MCMC steps', default = 200)
    parser.add_argument('-b', dest = 'burn', type = int, metavar = 'burn',
                        help = 'number of MCMC steps to discard as burn-in', default = 100)
    parser.add_argument('-u', dest = 'unres', action = 'store_true',
                        help = 'include a component of unresolved flux in the model')
    parser.add_argument('-t', dest = 'testres', action = 'store_true',
                        help = 'test for a resolved disc and skip if apparently not present')

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

    #model parameters
    alpha = args.alpha
    include_unres = args.unres

    #make high-resolution model with (hires_scale*hires_scale) sub-pixels per PACS pixel
    hires_scale = args.model_scale

    #where to save the results
    savepath = args.output

    #should we use PSF subtraction to test for a resolved disc?
    test = args.testres

    return (nwalkers, nsteps, burn, name_image, name_psf, dist, stellarflux,
            alpha, include_unres, hires_scale, savepath, name, test)


def run(nwalkers, nsteps, burn, name_image, name_psf, dist, stellarflux,
        alpha, include_unres, hires_scale, savepath, name, test):
    """Fit one image and save the output."""

    #load in image file and extract pixel scale
    with fits.open(name_image) as image_datafile:
        #extract image from FITS file
        image_data = image_datafile[1].data * 1000 #convert to mJy/pixel

        pfov = image_datafile[1].header['CDELT2'] * 3600 #pixel FOV in arcsec
        wav = int(image_datafile[0].header['WAVELNTH']) #wavelength of observations

        #extract the obsid
        obsid = image_datafile[0].header['OBSID001']

        #if a star name wasn't provided, use the target name from the observation header
        if name == '':
            name = image_datafile[0].header['OBJECT']

        image_angle = image_datafile[0].header['POSANGLE']

    image_data[np.isnan(image_data)] = 0

    aupp = pfov * dist #au per pixel

    #factors to correct for flux lost during high-pass filtering (see Kennedy et al. 2012)
    if wav == 70:
        flux_factor = 1.16
    elif wav == 100:
        flux_factor = 1.19

    #refuse to fit 160 micron data (70/100 is always available and generally at higher S/N)
    else:
        raise Exception("Please provide a 70 or 100 micron image.")

    #put the star name, obsid and wavelength together into an annotation for the image plot
    annotation = '\n'.join([f'PACS {wav} μm image', f'obsid: {obsid}', name])

    #scale up uncertainties since noise is correlated
    natural_pixsize = 3.2 #for PACS 70/100 micron images
    uncert_scale = correlated_noise_factor(natural_pixsize, pfov)

    #extract coverage level, so that we can estimate the rms flux using a suitable region of the sky
    cov = fits.getdata(name_image, 'coverage')
    cov[np.isnan(cov)] = 0

    #in portion of image with low coverage (i.e. towards edges), there are high levels of noise
    cov_lower_bound = 0.6 * np.max(cov)

    #find the coordinates of the star, assuming it's at the brightest pixel within
    #sky_separation_threshold arcsec of the image centre
    sky_separation_threshold = 14
    reference_indices = find_brightest(image_data, sky_separation_threshold, pfov)

    #find stellocentric distance to each pixel, so that we can cut out the
    #star (& disc) for the purposes of calculating the rms flux
    sky_separation = projected_sep_array(image_data.shape, reference_indices, pfov)


    #estimate the background rms flux
    rms = estimate_background(image_data[(cov > cov_lower_bound) & (sky_separation > sky_separation_threshold)])

    '''
    fig,ax = plt.subplots()
    ax.imshow(image_data, interpolation='none')
    ax.contour(cov, levels=np.array([cov_lower_bound]))
    ax.contour(sky_separation, levels=np.array([sky_separation_threshold]))
    ax.contour(image_data, levels=np.array([2*rms, 3*rms]))
    plt.show()
    '''


    #only interested in the star (& disc) at the centre of the image,
    #so cut out that part to save time
    img_boxscale = 13 #recall that the cutout will have dimension (2 * img_boxscale + 1)
    image_data = crop_image(image_data, reference_indices, img_boxscale)

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
    psf_boxscale = img_boxscale
    angle = psf_angle - image_angle
    psf_data = rotate(psf_data, angle)
    psf_data = crop_image(psf_data, find_brightest(psf_data, sky_separation_threshold, pfov), psf_boxscale)
    psf_data /= np.sum(psf_data)

    #rebin PSF to pixel scale of high-resolution model, then re-normalize
    psf_data_hires = congrid(psf_data, [i * hires_scale for i in psf_data.shape], method = 'linear', centre = False, minusone = True)
    psf_data_hires /= np.sum(psf_data_hires)

    #before starting to save output, remove any old files
    if os.path.exists(savepath):
        shutil.rmtree(savepath)

    os.makedirs(savepath)


    #######################################################
    '''
    ptest=[100, 100, 0, 0, 200, 300, 0, 40]
    print(np.unravel_index(image_data.argmax(), image_data.shape))
    print(np.unravel_index(psf_data.argmax(), psf_data.shape))
    model_unconvolved = model_hires(ptest, image_data.shape,
                                    np.unravel_index(image_data.argmax(), image_data.shape),
                                    aupp, hires_scale, alpha, include_unres, stellarflux,
                                    flux_factor, include_central = True)

    model_unconvolved[model_unconvolved <= 0] = np.amin(model_unconvolved[model_unconvolved > 0]) / 2
    fig,ax=plt.subplots()
    plot_image(ax, model_unconvolved, pfov, scale = hires_scale, annotation = 'High-resolution model',
               log = True, scalebar = True, dist = dist)
    #plt.pcolormesh(np.log10(model_unconvolved), edgecolors='w',linewidth = 0.25)
    plt.show()

    model_unconvolved = distance_array(ptest, image_data.shape,
                                    np.unravel_index(image_data.argmax(), image_data.shape),
                                    aupp, hires_scale, include_unres)

    fig,ax=plt.subplots()
    plot_image(ax, model_unconvolved, pfov, scale = hires_scale, annotation = 'Distance array',
               log = False, scalebar = True, dist = dist)
    plt.show()
    '''
    #######################################################


    #if requested, first check whether the image is consistent with a PSF and skip the fit if possible
    psfsub = best_psf_subtraction(image_data, psf_data, rms * uncert_scale)

    if test:
        sig, is_noise = consistent_gaussian(psfsub.flatten())

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
            plt.close(fig)

            #for consistency, save a pickle indicating that no disc was resolved
            save_params(savepath, False)

            #plt.show()

            return

        else:
            print(f"The PSF subtraction is not consistent with Gaussian noise at the {sig:.0f}% level."
                  " There may be a resolved disc here. Performing disc fit.")


    #find best-fitting parameters using differential evolution, which searches for the
    #global minimum within the parameter ranges specified by the arguments.
    #format is [<funres,> ftot, x0, y0, r1, r2, inc, theta]

    #NOTE: to do - parametrize the limits on r1, r2 and inc.
    search_space = [(0, 10000), (0, 10000), (-300, 300), (-300, 300),
                    (5, 1000), (5, 1000), (0, 85), (0, 180)]

    pnames = [r'$F_\mathrm{unres}\ /\ \mathrm{mJy}$', r'$F_\mathrm{res}\ /\ \mathrm{mJy}$',
              r'$x_0\ /\ \mathrm{au}$', r'$y_0\ /\ \mathrm{au}$',
              r'$r_1\ /\ \mathrm{au}$', r'$r_2\ /\ \mathrm{au}$',
              r'$i\ /\ \mathrm{deg}$', r'$\theta\ /\ \mathrm{deg}$'] #parameter names for plot labels

    #if we're not including an unresolved flux parameter, remove the first element
    #of the parameter list
    if not include_unres:
        search_space.pop(0)
        pnames.pop(0)
        pnames[0] = r'$F_\mathrm{disc}\ /\ \mathrm{mJy}$'

    popsize = 20 #population size for differential evolution
    maxiter = 50 #number of differential evolution iterations

    print("Finding a suitable initial model...")

    pbar = tqdm.tqdm(total = maxiter)

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
    fig, ax = plt.subplots(ndim, figsize = (12, 12), sharex = True)
    chain = sampler.get_chain()

    for i in range(ndim):
        ax[i].plot(chain[:, :, i], c = 'k', alpha = 0.3)

        #shade the burn-in period
        ax[i].axvspan(0, burn - 0.5, alpha = 0.1, color = 'k')

        ax[i].xaxis.set_major_locator(MaxNLocator(integer = True))
        ax[i].set_xlim(0, nsteps - 1)
        ax[i].set_ylabel(pnames[i])

    ax[-1].set_xlabel("Step number")

    plt.tight_layout()
    fig.savefig(savepath + '/chains.pdf')
    plt.close(fig)


    print("Exporting corner plot...")

    #make the corner plot
    fig = corner.corner(samples, quantiles = [0.16, 0.84], labels = pnames, show_titles = True, title_fmt = ".0f")
    fig.savefig(savepath + '/corner.pdf')
    plt.close(fig)


    #get max-likelihood model parameters (no need to exclude burn-in for this, just want the best fit)
    max_likelihood = sampler.flatchain[np.argmax(sampler.flatlnprobability), :]

    #get median and 16th/84th percentile parameters
    median = np.median(samples, axis = 0)
    lower_uncertainty = median - np.percentile(samples, 16, axis = 0)
    upper_uncertainty = np.percentile(samples, 84, axis = 0) - median


    # Now make a four-panel image: [data, psf subtraction, high-res max-likelihood model, residuals]
    print("Exporting image of best-fit model...")

    model = synthetic_obs(max_likelihood, psf_data_hires, image_data.shape,
                          aupp, hires_scale, alpha, include_unres, stellarflux, flux_factor)

    #don't include the central bright pixel for display purposes
    model_unconvolved = model_hires(max_likelihood, image_data.shape,
                                    aupp, hires_scale, alpha, include_unres, stellarflux,
                                    flux_factor, include_central = False)

    residual = image_data - model

    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (18, 6), sharey = True)

    #first plot the PACS data
    plot_image(ax[0], image_data, pfov, ylabel = True, annotation = annotation)

    #then the PSF subtraction
    plot_image(ax[1], psfsub, pfov, annotation = 'PSF subtraction')
    plot_contours(ax[1], psfsub, pfov, rms)

    #now the high-res model; set zero pixels to small amount (half the smallest pixel) as plotting on log scale
    model_unconvolved[model_unconvolved <= 0] = np.amin(model_unconvolved[model_unconvolved > 0]) / 2
    plot_image(ax[2], model_unconvolved, pfov, scale = hires_scale, annotation = 'High-resolution model',
               log = True, scalebar = True, dist = dist)

    #finally, the mopdel residuals
    plot_image(ax[3], residual, pfov, annotation = 'Residuals')
    plot_contours(ax[3], residual, pfov, rms)

    plt.tight_layout()
    fig.savefig(savepath + '/image_model.png', dpi = 150)
    plt.show()
    plt.close(fig)


    #check whether the model appears to be a good fit
    sig, is_noise = consistent_gaussian(residual.flatten())

    if is_noise:
        print(f"The residuals are consistent with Gaussian noise at the {sig:.0f}% significance level."
              " The disc model appears to explain the data well.")
    else:
        print(f"The residuals are not consistent with Gaussian noise at the {sig:.0f}% significance level."
              " You may wish to check the residuals for issues.")


    #finally, save some important parameters in a pickle
    save_params(savepath, True, include_unres, max_likelihood, median, lower_uncertainty, upper_uncertainty,
                is_noise)


if __name__ == "__main__":
    run(*parse_args())
    
