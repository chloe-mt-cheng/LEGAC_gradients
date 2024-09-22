"""Usage: calculate_seeing.py [-h][--path=<arg>][--target=<arg>][--directory=<arg>][--pixscale=<arg>][--output_name=<arg>][--input_name=<arg>]

Examples:
	Path: e.g. input --path='/path/to/main/directory'
	Target: e.g. input --target='M5_172669'
	Directory: e.g. input --directory='temp'
	Pixel scale: e.g. input --pixscale=0.205
	Output name: e.g. input --output_name='M5_172669_seeing'
	Input name: e.g. input --input_name='M5_172669_seeing'

-h  Help file
--path=<arg>  Path to where you want to save files
--target=<arg> Name of galaxy
--directory=<arg> Directory where galfit files are 
--pixscale=<arg> Pixelscale of galfit image
--output_name=<arg> Name of galfit output file
--input_name=<arg> Name of galfit input file

"""

import numpy as np
import pandas as pd
from docopt import docopt
import glob
import matplotlib.pyplot as plt
import os
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
import astropy.units as u
from astropy.convolution import Moffat2DKernel, convolve
from photutils.aperture import aperture_photometry, RectangularAperture, CircularAperture, EllipticalAperture
from scipy.optimize import curve_fit
from scipy.integrate import quad
import math
import itertools

# Read in LEGA-C catalogue for image parameters
my_cat = pd.read_csv('moffat_cat_final.csv', header=0, index_col=0) 
LEGAC_cat_file = fits.open('legac_dr3_cat.fits')
LEGAC_cat_head = LEGAC_cat_file[1].header
LEGAC_cat = LEGAC_cat_file[1].data
LEGAC_cat_file.close()

LEGAC_df = pd.DataFrame({'MASK': LEGAC_cat['MASK'].byteswap().newbyteorder(), 'ID': LEGAC_cat['ID'].byteswap().newbyteorder(), 
                         'RA': LEGAC_cat['RA'].byteswap().newbyteorder(), 'DEC': LEGAC_cat['DEC'].byteswap().newbyteorder()})

LEGAC_spect_ids = []
for i in range(len(LEGAC_df)):
    LEGAC_spect_ids.append('M' + str(int(LEGAC_df['MASK'].iloc()[i])) + '_' + str(int(LEGAC_df['ID'].iloc()[i])))

LEGAC_final_df = pd.DataFrame({'SPECT_ID': LEGAC_spect_ids, 
                               'MAG': LEGAC_cat['SERSIC_MAG'].byteswap().newbyteorder(),
                               'RE': LEGAC_cat['SERSIC_RE'].byteswap().newbyteorder(), 
                               'N': LEGAC_cat['SERSIC_N'].byteswap().newbyteorder(),
                               'Q': LEGAC_cat['SERSIC_Q'].byteswap().newbyteorder(),
                               'PA': LEGAC_cat['SERSIC_PA'].byteswap().newbyteorder()})

my_sample_df = LEGAC_final_df.merge(my_cat, on='SPECT_ID').drop_duplicates(subset='SPECT_ID')

def moffat(x, x0, amplitude, gamma, alpha):
    """From van Houdt+2021, same as in astropy Moffat1D.
    """
    return amplitude*(1 + ((x - x0)**2/gamma**2))**-alpha

def spectral_profile(target):
    """Return the 1D profile measured from the spectrum.
    
    Parameters
    ----------
    target : str
        Target name (mask + ID, i.e. 'M5_172669')
    
    Returns
    -------
    norm_profile : tuple
    	1D flux profile from spectrum
    noise_profile : tuple
    	1D variance profile from spectrum
    total_spectrum_flux : float
    	Total amount of flux in the pixels of the 2D spectrum
    """
    
    #Import spectrum and weight map
    mask, obj = target.split('_')[0], target.split('_')[1]
    spec2d_file = fits.open('/2Dspectra/legac_' + mask + '_v3.11_spec2d_' + obj + '.fits')
    flux2D = spec2d_file[0].data
    spec2d_file.close()
    
    wht2d_file = fits.open('/2Dspectra/legac_' + mask + '_v3.11_wht2d_' + obj + '.fits')
    wht2D = wht2d_file[0].data
    wht2d_file.close()
    
    # Make sky spectrum and skyline mask 
    # Identify edge columns that == 0
    midrow = int(flux2D.shape[0]/2) # This is the number of rows/2
    mask1D = flux2D[midrow]!=0
    
    # Remove 0's from weight spectrum
    num_wht = np.copy(wht2D)
    for i in range(len(num_wht)):
        for j in range(len(num_wht[i])):
            if num_wht[i][j] == 0.0:
                num_wht[i][j] = np.nan
    sky2D = 1/num_wht #this is the variance
    
    # Make sky spectrum by collapsing over spatial axis
    sky1D = np.nansum(sky2D, axis=0)
    
    # Mask out the edges and use percentiles to find where we should mask lines
    skycut = np.nanpercentile(sky1D[mask1D][5:-5], 85)
    
    # Make profile (mask skylines)
    # Make a 2D mask that masks out the edges of the spectrum and where the sky is very bright
    mask2D = np.array([mask1D & (sky1D < skycut)]*flux2D.shape[0])
    profile = np.nansum(flux2D*mask2D, axis=1) # Profile is spatial, so you collapse over wavelength
    noise_profile = np.nansum(sky2D*mask2D, axis=1) #do you normalize this as well?
    
    #Mask negative fluxes
    mask_profile = profile*(profile > 0)
    mask_noise = noise_profile*(profile > 0) #Do you mask the noise spectrum as well?  I think so
    
    return mask_profile, mask_noise
    
def img_centre(img_length):
    """Return the centre of the image in pixels (as defined by galfit).
    
    Parameters
    ----------
    img_length : float
    	Length of the image side (length of the spectral profile)
    	
    Returns
    -------
    centre : float
    	Centre of the image in pixels
    """

    if img_length%2 == 1: # odd length
        centre = np.median(np.linspace(1, img_length, img_length))
    else: # even length
        centre = img_length/2 + 1
        
    return centre
    
def generate_psf(gamma, alpha):
    """Return the PSF (2D Moffat by which the image will be convolved).
    
    Parameters
    ----------
    gamma : float
    	Core width of the Moffat 
    alpha : float
        Power index of the Moffat (controls wing shape)
    img_length : float
    	Size of the image (length of the spectral profile)
    	
    Returns
    -------
    psf : tuple
    	2D image array containing the Moffat PSF
    """

    #Generate a 2D Gaussian PSF
    moffat_psf = Moffat2DKernel(gamma, alpha)
    psf = moffat_psf.array
    
    return psf
    
# Something weird is happening when you convolve within galfit
# So I will leave this out and convolve the image myself after the model is created
def galfit_input(targ, directory, img_box, img_ctr, pixscale, output_name, input_name):
    """Write a galfit input file.
    
    Parameters
    ----------
    targ : str
    	Target name (mask + ID, i.e. 'M5_172669')
    directory : str
    	Directory where you want the input files to be stored
    img_box : float
    	Length of the image (length of the profile)
    img_ctr : float
    	Centre of the image in pixels
    pixscale : float
        Pixelscale [arcsec/pixel]
    output_name : float
    	Name of the galfit output file (no extension)
    input_name : float
    	Name of the galfit input file (no extension)
    	
    Returns
    -------
    None
    """

    targ_df = my_sample_df[my_sample_df['SPECT_ID'] == targ]
    keys = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 'I)', 'J)', 'K)', 'O)', 'P)', '0)', '1)', '3)', '4)', '5)', '9)', '10)', 'Z)']
    vals = ['none  #Input data image (FITS file)', 
            output_name + '_galfit.fits  #Output data image block', 
            'none  #Sigma image name', 
            'none #Input PSF image and diffusion kernel', 
            'none  #PSF fine sampling factor', 
            'none  #Bad pixel mask', 
            'none  #File with parameter constraints', 
            '1  ' + str(img_box) + '  1  ' + str(img_box) + '  #Image region to fix (xmin xmax ymin ymax)', 
            'none #Size of the convolution box (x y)', 
            '0  #Magnitude photometric zeropoint', 
            'none #Plate scale (dx dy)',
            'regular  #Display type', 
            '1  #Options (1 = make model only)', 
            'sersic  #Object type', 
            str(img_ctr) + '  ' + str(img_ctr) + '  0  0  #Pixel position (x y)', 
            str(targ_df['MAG'].values[0]) + '  0  #Total magnitude', 
            str(targ_df['RE'].values[0]/float(pixscale)) + '  0  #Re (pixels)', 
            str(targ_df['N'].values[0]) + '  0  #Sersic exponent (n)', 
            str(targ_df['Q'].values[0]) + '  0  #Axis ratio (b/a)', 
            str(targ_df['PA'].values[0]) + '  0  #Position angle (PA, degrees up = 0, left = 90)', 
            '0  #Skip this model in output image? (yes = 1, no=0)']

    input_dict = dict(zip(keys, vals))
    input_df = pd.Series(input_dict, index=input_dict.keys())
    input_df.to_csv('/galfit/%s/%s.input' %(directory, input_name), sep='\t', header=None)
    
def galfit_model(directory, input_name):
    """Run galfit from OS and put the output files in a directory.
    
    Parameters
    ----------
    directory : str
        Desired output directory
    input_name : str
        galfit input file name
        
    Returns
    -------
    None
    """
    
    os.system('/galfit/galfit /galfit/%s/%s.input >/dev/null 2>&1' %(directory, input_name))
    os.system('mv *_galfit.fits /galfit_imgs/%s/' %directory)
    
def moffat_percentile(model, popt_moffat, percentile):
    
    total = quad(moffat, 0, len(model), args=(popt_moffat[0], popt_moffat[1], popt_moffat[2], popt_moffat[3]))[0]
    percentile_val = (percentile/100)*total
    b_vals = np.linspace(0, len(model), 10000)

    for i in range(len(b_vals)):
        integral = quad(moffat, a=0, b=b_vals[i], args=(popt_moffat[0], popt_moffat[1], popt_moffat[2], popt_moffat[3]))[0]
        if integral > percentile_val:
            p = b_vals[i-1]
            break
            
    return p
    
def image_profile(directory, galfit_model, psf, img_length, pixscale, spec_profile, spec_noise):
    """Return the flux profile from the model image.
   
    Parameters
    ----------
    directory : str
       Directory where your output files are kept
    galfit_model : str
       Filename of desired galfit model
    psf : tuple
        PSF image
    img_length : float
        Size of the image
    pixscale : float
        Pixelscale [arcsec/pixel]
    spec_profile : tuple
        Spectral profile
    spec_noise : tuple
        Noise from spectral profile
    
    Returns
    -------
    norm_profile : tuple
        Normalized profile from the galfit model image
    norm_y : tuple
        Normalized profile image over extraction region
    spec_y : 
        Spectral profile over extraction region
    noise_y : 
        Variance profile over extraction region
   """
    
    # Read in galfit image
    galfit_file = fits.open('/galfit_imgs/%s/%s_galfit.fits' %(directory, galfit_model))
    galfit_img = galfit_file[0].data
    galfit_file.close()
    
    # Convolve the image with the PSF outside of galfit
    if np.nansum(psf) == 0:
        convolved_img = convolve(galfit_img, psf, nan_treatment='fill')
    else:
        convolved_img = convolve(galfit_img, psf, nan_treatment='interpolate')
    
    # Define slit aperture and mask
    # for image centre, need to subtract 1 because of the 1 counting in galfit
    aperture = RectangularAperture((img_centre(img_length) - 1, img_centre(img_length) - 1), w=1/float(pixscale), h=img_length, theta=Angle(0*u.deg))
    aperture_mask = aperture.to_mask()
    
    # Mask the image
    weighted_img = aperture_mask.multiply(convolved_img)
    
    # Collapse along the slit width 
    profile_img = np.nansum(weighted_img, axis=1)
    
    # Find the normalization factor and normalize the image
    #Also try over region that you actually examine (i.e. no wings)
    #Fit the moffat profile to the spectrum
    xfit = np.arange(0, len(spec_profile), 1)
    yfit = spec_profile*(spec_profile > 0)
    popt, _ = curve_fit(moffat, xfit, yfit,p0=[np.nanargmax(spec_profile), np.nanmax(spec_profile), 1.5, 1.1])
    
    #Calculate the percentiles of the fit to the spectral profile
    percentile_3 = moffat_percentile(moffat(xfit, *popt), popt, 3)
    percentile_97 = moffat_percentile(moffat(xfit, *popt), popt, 97)

    #Fractional thing like in bins
    spec_y = spec_profile*(spec_profile > 0)
    img_y = profile_img*(spec_profile > 0)
    noise_y = spec_noise*(spec_profile > 0)
    
    spec_y[math.floor(percentile_3)] = spec_y[math.floor(percentile_3)]*(percentile_3 - math.floor(percentile_3))
    spec_y[:math.floor(percentile_3)] = 0.0
    spec_y[math.ceil(percentile_97)] = spec_y[math.ceil(percentile_97)]*(percentile_97 - math.floor(percentile_97))
    spec_y[math.ceil(percentile_97)+1:] = 0.0

    img_y[math.floor(percentile_3)] = img_y[math.floor(percentile_3)]*(percentile_3 - math.floor(percentile_3))
    img_y[:math.floor(percentile_3)] = 0.0
    img_y[math.ceil(percentile_97)] = img_y[math.ceil(percentile_97)]*(percentile_97 - math.floor(percentile_97))
    img_y[math.ceil(percentile_97)+1:] = 0.0
    
    #Also do the noise (need for chisquare)
    noise_y[math.floor(percentile_3)] = noise_y[math.floor(percentile_3)]*(percentile_3 - math.floor(percentile_3))
    noise_y[:math.floor(percentile_3)] = 0.0
    noise_y[math.ceil(percentile_97)] = noise_y[math.ceil(percentile_97)]*(percentile_97 - math.floor(percentile_97))
    noise_y[math.ceil(percentile_97)+1:] = 0.0

    norm_factor = np.nansum(spec_y*img_y)/np.nansum(img_y**2)
    norm_profile = profile_img * norm_factor
    
    #Get norm profile over extraction region for chisquare later
    norm_y = norm_profile*(spec_profile > 0)
    norm_y[math.floor(percentile_3)] = norm_y[math.floor(percentile_3)]*(percentile_3 - math.floor(percentile_3))
    norm_y[:math.floor(percentile_3)] = 0.0
    norm_y[math.ceil(percentile_97)] = norm_y[math.ceil(percentile_97)]*(percentile_97 - math.floor(percentile_97))
    norm_y[math.ceil(percentile_97)+1:] = 0.0
    
    return norm_profile, norm_y, spec_y, noise_y
    
def chisq_stat(flux, model, var):
    """Return the goodness of fit chi squared statistic.
    
    Parameters
    ----------
    flux : tuple
        Data flux over extraction region
    model : tuple
        Model flux over extraction region
    var : tuple
        Variance of the data over extraction region
        
    Return
    ------
    np.nansum(chi) : float
        Reduced chisq statistic
    """
    
    # Calculate chisquare for extraction area (avoid weird artefacts)
    chi = (np.array(flux) - np.array(model))**2/np.array(var)
    for i in range(len(chi)):
        if ~np.isfinite(chi[i]):
            chi[i] = np.nan
            
    return np.nansum(chi)
    
def FWHM_moffat(gamma, alpha):
    return 2*gamma*np.sqrt(2**(1/alpha) - 1)
    
def measure_seeing(path, target, directory, pixscale, output_name, input_name):
    """Return the best-fit seeing for a galaxy.
    
    Parameters
    ----------
    path : str
        Path where you want to save the files
    target : str
        Name of the galaxy
    directory : str
        Galfit directory
    pixscale : float
        Pixel scale in arcseconds/pixel
    output_name : str
        Galfit output file name
    input_name : str
        Galfit input file name
        
    Returns
    -------
    best_seeing : float
        Best-fit seeing value (in PIXELS)
    param_grid[np.argmin(chisq_grid)] : tuple
        Best-fit gamma and alpha parameters
    profile_img_grid[np.argmin(chisq_grid)] : tuple
        Best-fit profile from image
    spec_profile : tuple
        Spectral profile
    """
    
    #List of possible gamma and alpha values
    params = np.linspace(0.1, 7, 20)

    #Generate all possible two-element combinations
    #Convert the resulting iterator to a list
    #(gamma, alpha)
    param_grid = list(itertools.permutations(params, 2))

    #Run through grid
    spec_profile, spec_noise = spectral_profile(target)
    img_size = len(spec_profile)

    #Generate the psfs
    psf_grid = []
    for i in range(len(param_grid)):
        psf_grid.append(generate_psf(param_grid[i][0], param_grid[i][1]))
    
    # Generate the galfit model 
    galfit_input(target, directory, img_size, img_centre(img_size), pixscale, output_name, input_name)
    galfit_model(directory, input_name)
    
    # Get image profiles
    profile_img_grid = []
    norm_y_grid = []
    spec_y_grid = []
    noise_y_grid = []
    for i in range(len(psf_grid)):
        img_profile_calc = image_profile(directory, input_name, psf_grid[i], img_size, pixscale, spec_profile, spec_noise)
        profile_img_grid.append(img_profile_calc[0])
        norm_y_grid.append(img_profile_calc[1])
        spec_y_grid.append(img_profile_calc[2])
        noise_y_grid.append(img_profile_calc[3])
    
    # Compare each image profile to spectrum 
    chisq_grid = []
    for i in range(len(profile_img_grid)):
        chisq_grid.append(chisq_stat(spec_y_grid[i], norm_y_grid[i], noise_y_grid[i]))
        
    best_seeing = FWHM_moffat(param_grid[np.argmin(chisq_grid)][0], param_grid[np.argmin(chisq_grid)][1])
    
    #Write to file
    seeing_df = pd.DataFrame({'seeing': best_seeing, 'gamma': param_grid[np.argmin(chisq_grid)][0], 'alpha': param_grid[np.argmin(chisq_grid)][1], 'img_profile': profile_img_grid[np.argmin(chisq_grid)], 'spec_profile': spec_profile})
    seeing_df.to_csv(path + '/seeing_files/%s_seeing.dat' %target, index=None, sep='\t')
        
    return best_seeing, param_grid[np.argmin(chisq_grid)], profile_img_grid[np.argmin(chisq_grid)], spec_profile
    
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	best_seeing, best_params, best_img_profile, spec_profile = measure_seeing(arguments['--path'], arguments['--target'], arguments['--directory'], arguments['--pixscale'], arguments['--output_name'], arguments['--input_name'])