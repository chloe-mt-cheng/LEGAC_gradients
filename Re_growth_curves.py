"""Usage: Re_growth_curves.py [-h][--path=<arg>][--target=<arg>][--best_seeing=<arg>][--directory=<arg>][--pixscale=<arg>][--output_name=<arg>][--input_name=<arg>][--img_factor=<arg>]

Examples:
	Path: e.g. input --path='/Users/chloecheng/Documents/LEGAC_resolved'
	Target: e.g. input --target='M5_172669'
	Best seeing: e.g. input --best_seeing=1.8
	Directory: e.g. input --directory='temp'
	Pixel scale: e.g. input --pixscale=0.1
	Output name: e.g. input --output_name='M5_172669_gc'
	Input name: e.g. input --input_name='M5_172669_gc'
	Image factor: e.g. input --img_factor=51

-h  Help file
--path=<arg>  Path to where you want to save files
--target=<arg> Name of galaxy
--best_seeing=<arg> FWHM of best-fit seeing
--directory=<arg> Directory where galfit files are 
--pixscale=<arg> Pixelscale of galfit image
--output_name=<arg> Name of galfit output file
--input_name=<arg> Name of galfit input file
--img_factor=<arg> Factor by which to increase the image size (odd number)

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
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.aperture import aperture_photometry, RectangularAperture, CircularAperture, EllipticalAperture
from scipy.optimize import curve_fit

# Read in LEGA-C catalogue for image parameters
my_cat = pd.read_csv('zuvj_LEGAC_cat.csv', header=0, index_col=0)
LEGAC_cat_file = fits.open('/Users/chloecheng/Documents/LEGAC_resolved/legac_dr3_cat.fits')
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
    spec2d_file = fits.open('/Users/chloecheng/Documents/LEGAC_resolved/2Dspectra/legac_' + mask + '_v3.11_spec2d_' + obj + '.fits')
    flux2D = spec2d_file[0].data
    spec2d_file.close()
    
    wht2d_file = fits.open('/Users/chloecheng/Documents/LEGAC_resolved/2Dspectra/legac_' + mask + '_v3.11_wht2d_' + obj + '.fits')
    noise2D = wht2d_file[0].data
    wht2d_file.close()
    
    # Make sky spectrum and skyline mask 
    # Identify edge columns that == 0
    midrow = int(flux2D.shape[0]/2) # This is the number of rows/2
    mask1D = flux2D[midrow]!=0
    
    # Remove 0's from weight spectrum
    num_wht = np.copy(noise2D)
    for i in range(len(num_wht)):
        for j in range(len(num_wht[i])):
            if num_wht[i][j] == 0.0:
                num_wht[i][j] = np.nan
    sky2D = 1/num_wht
    
    # Make sky spectrum by collapsing over spatial axis
    sky1D = np.nansum(sky2D, axis=0)
    
    # Mask out the edges and use percentiles to find where we should mask lines
    skycut = np.nanpercentile(sky1D[mask1D][5:-5], 85)
    
    # Make profile (mask skylines)
    # Make a 2D mask that masks out the edges of the spectrum and where the sky is very bright
    mask2D = np.array([mask1D & (sky1D < skycut)]*flux2D.shape[0])
    profile = np.nansum(flux2D*mask2D, axis=1) # Profile is spatial, so you collapse over wavelength
    noise_profile = np.nansum(noise2D*mask2D, axis=1) #do you normalize this as well?
    
    #Mask negative fluxes
    mask_profile = profile*(profile > 0)
    mask_noise = noise_profile*(profile > 0) #Do you mask the noise spectrum as well?  I think so
    
    # Find total flux from all pixels in the spectrum
    # For normalization later
    total_spectrum_flux = np.nansum(flux2D*(flux2D > 0)) #not 100% sure if this is the right way to do it 
    
    return mask_profile, mask_noise, total_spectrum_flux
    
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
    
def generate_psf(seeing, img_length):
    """Return the PSF (2D Gaussian by which the image will be convolved).
    
    Parameters
    ----------
    seeing : float
    	Width of the Gaussian (seeing by which to convolve)
    img_length : float
    	Size of the image (length of the spectral profile)
    	
    Returns
    -------
    psf : tuple
    	2D image array containing the Gaussian PSF
    """

    #Generate a 2D Gaussian PSF
    gaussian_psf = Gaussian2DKernel(seeing)
    psf = gaussian_psf.array
    
    return psf
    
# Something weird is happening when you convolve within galfit
# So I will leave this out and convolve the image myself after the model is created
def galfit_input(targ, directory, img_box, img_ctr, seeing, pixscale, output_name, input_name):
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
    seeing : float
    	Width of the Gaussian PSF (seeing by which you will convolve)
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
            str(targ_df['RE'].values[0]/pixscale) + '  0  #Re (pixels)', 
            str(targ_df['N'].values[0]) + '  0  #Sersic exponent (n)', 
            str(targ_df['Q'].values[0]) + '  0  #Axis ratio (b/a)', 
            str(targ_df['PA'].values[0]) + '  0  #Position angle (PA, degrees up = 0, left = 90)', 
            '0  #Skip this model in output image? (yes = 1, no=0)']

    input_dict = dict(zip(keys, vals))
    input_df = pd.Series(input_dict, index=input_dict.keys())
    input_df.to_csv('/Users/chloecheng/packages/galfit/%s/%s.input' %(directory, input_name), sep='\t', header=None)
    
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
    
    os.system('/Users/chloecheng/packages/galfit/galfit /Users/chloecheng/packages/galfit/%s/%s.input >/dev/null 2>&1' %(directory, input_name))
    os.system('mv /Users/chloecheng/Documents/LEGAC_resolved/%s_galfit.fits /Users/chloecheng/Documents/LEGAC_resolved/galfit_imgs/%s/' %(input_name, directory))
    

def rescale(data):
    """Return data rescaled from 0 to 1 on the y-axis.
    
    Parameters 
    ----------
    data : tuple
        Input data
    
    Returns
    -------
    (data - np.min(data))/(np.max(data) - np.min(data)) : tuple
        Rescaled data
    """
    
    return (data - np.min(data))/(np.max(data) - np.min(data))
    
def find_nearest(array, value):
    """Return the index and the value of the value nearest to a desired value in an array.
    
    Parameters
    ----------
    array : tuple
        Input data
    value : float
        Desired value
        
    Returns
    -------
    idx : int
        Index nearest to the desired value
    array[idx] : float
        Value nearest to the desired value
    """
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return idx, array[idx]

def calculate_Re(path, target, best_seeing, directory, pixscale, output_name, input_name, img_factor):
    """Return the Re in pixels and the growth curve.
    
    Parameters
    ----------
    path : str
        Path to where you want to save files
    target : str
        Name of galaxy
    best_seeing : float
        FWHM of best-fit seeing
    directory : str
        Directory where galfit files are 
    pixscale : float
        Pixelscale of galfit image
    output_name : str
        Name of galfit output file
    input_name : str
        Name of galfit input file
    img_factor : int
        Factor by which to increase the image size (odd number)
        
    Returns
    -------
    Re : float
        Re in pixels
    circle_radii : tuple
        List of radii of circular apertures
    rescale(circle_sums) : tuple
        Rescaled growth curve
    """

    # Get the profile from the spectrum
    profile_spectrum, noise_spec, total_spectrum_flux = spectral_profile(target)

    # Generate a higher-resolution and larger image and a PSF with best-fit seeing
    img_size = int(len(profile_spectrum)*int(img_factor)) 
    img_ctr = img_centre(int(img_size))
    big_psf = generate_psf(float(best_seeing), int(img_size))
    galfit_input(target, directory, int(img_size), img_ctr, float(best_seeing), float(pixscale), output_name, input_name)
    galfit_model(directory, input_name)
    
    # Read in this model
    galfit_file = fits.open('/Users/chloecheng/Documents/LEGAC_resolved/galfit_imgs/%s/%s_galfit.fits' %(directory, output_name))
    galfit_img = galfit_file[0].data
    galfit_file.close()
    #plt.imshow(galfit_img, aspect='auto', origin='lower')
    
    # Convolve the image with the PSF outside of galfit
    if np.nansum(big_psf) == 0:
        convolved_img = convolve(galfit_img, big_psf, nan_treatment='fill')
    else:
        convolved_img = convolve(galfit_img, big_psf, nan_treatment='interpolate')
        
    #plt.figure()
    #plt.imshow(convolved_img, aspect='auto', origin='lower')
    
    # Make circular apertures of increasing radius
    circle_radii = np.linspace(1, int(img_size), int(img_factor)*1333) #Got this number from my calculations in growth_curve_troubleshoot.ipynb 
    circle_apertures = []
    for i in range(len(circle_radii)):
        if img_size % 2 == 0:
            circle_apertures.append(CircularAperture((img_ctr - 2, img_ctr - 2), circle_radii[i])) # If side length is even, subtract 2 (I think this is still a counting thing)
        else:
            circle_apertures.append(CircularAperture((img_ctr - 1, img_ctr - 1), circle_radii[i]))
            
    # Perform aperture sums
    circle_sums = []
    for i in range(len(circle_apertures)):
        circle_sums.append(aperture_photometry(convolved_img, circle_apertures[i])['aperture_sum'])
    
    # Calculate Re
    Re = circle_radii[find_nearest(rescale(circle_sums), 0.5)[0]]
    
    #Write to file
    Re_df = pd.DataFrame({'Re': Re, 'radii': circle_radii, 'cumsum': rescale(circle_sums).flatten()})
    Re_df.to_csv(path + '/re_files/%s_Re.dat' %target, index=None, sep='\t')
    
    return Re, circle_radii, rescale(circle_sums)
    
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	Re_pix, radii_pix, cumsum = calculate_Re(arguments['--path'], arguments['--target'], arguments['--best_seeing'], arguments['--directory'], arguments['--pixscale'], arguments['--output_name'], arguments['--input_name'], arguments['--img_factor'])