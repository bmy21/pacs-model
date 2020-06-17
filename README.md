# pacs-model

Fit a debris disc model to a *Herschel* PACS image. The code here is an extension of that used to model the disc of HD 95698 in this paper: [2019MNRAS.488.3588Y](https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.3588Y/abstract).

## Usage

Fitting is performed by the script [pacs_model.py](pacs_model.py). You can either import the file and call the `run` function, or run the script from the command line and pass arguments that way. The only required argument is `name_image`, which should be a string pointing towards the FITS file containing the data that need fitting. The optional arguments allow you to control the details of the fit (e.g. model resolution, number of iterations), and to provide extra information about the system of interest that will make the results more informative (e.g. distance, expected stellar flux). For full details of the optional arguments, run `python pacs_model.py -h`.

For fitting a large number of images, [pacs_model_batch.py](pacs_model_batch.py) should be helpful. This script reads in a CSV file ([obs_path_list.csv](input/obs_path_list.csv)) containing information about the images to fit and calls `run` for each system. An appropriately formatted CSV can be produced with the help of the notebook [get_obs_paths.ipynb](get_obs_paths.ipynb). However, note that [pacs_model_batch.py](pacs_model_batch.py) could be easily modified to read in a CSV with less information and make use of the default argument values.

Note that the code models discs as having a single power-law radial surface brightness profile between an inner and outer radius. This is often a good approximation, but high signal-to-noise images of discs with more complicated radial profiles may require a more detailed model.


## Output

For each system modelled, [pacs_model.py](pacs_model.py) will save:

- **chains.pdf**: a plot showing the evolution of the MCMC walkers over time;
- **corner.pdf**: a plot showing the 2D projections of the posterior model parameter distributions;
- **image_model.png**: a plot showing the image, PSF-subtracted image, best-fitting model and residuals;
- **samples.pickle**: a pickle containing the MCMC samples (excluding burn-in) for future analysis;
- **params.pickle**: a pickle containing the key results of the fit.

Note that if `run` receives `test = True`, the code will first try to check whether the image is consistent with a point source, and skip the disc fit if so (see the function `consistent_gaussian` for details). In this case, image_model.png will only contain the first two panels, and the MCMC-related output won't be produced. 


## Implementation details

Disc parameters are sampled using the MCMC technique (via the [emcee](https://emcee.readthedocs.io/en/stable/) package). The disc model itself is a purely geometric one, where the sky is divided up into pixels (which can, and probably should, be smaller than the PACS pixels), and the physical distance from the star to the disc element at each pixel is calculated given the disc's proposed inclination and position angle. The model works well in general but breaks down for edge-on discs, and hence the inclination should be restricted to below 90 degrees. This is not too much of a limitation in practice: restricting inclination to 88 degrees still allows AU Mic, a known edge-on disc, to be well modelled.


## Future improvements

Some features that it would be beneficial to add in the future:

- An option to include more than one point source in the model, to account for any nearby companions or background objects. Since the code currently doesn't do this, what generally happens in multiple-source images is that the best-fitting model has a disc whose ansae pass through at least one of the sources. These 'discs' can sometimes fit the data well, but are unlikely to be the true explanation of what's going on!

- A full 3D disc model that doesn't require any restriction on inclination. This would involve generating a number of particles in the disc based on its parameters, then binning their positions in the sky plane to get the observed intensity. The main advantage of this would be better handling of high inclinations. However, the current model allows for much faster fitting and, as noted above, generally works sufficiently well to allow the disc parameters to be constrained.
