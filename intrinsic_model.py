"""Usage: intrinsic_model.py [-h][--path=<arg>][--FeH_m=<arg>][--targ=<arg>][--directory=<arg>][--subpix=<arg>][--core_radius=<arg>][--outskirt_radius=<arg>][--grid_size=<arg>]

Examples:
	Path: e.g. input --path='/Users/chloecheng/Documents/LEGAC_resolved'
	FeH gradient slope: e.g. input --FeH_m=-0.03
	Target: e.g. input --targ='M5_172669'
	Directory: e.g. input --directory='intrinsic_model'
	Sub pixels: e.g. input --directory=529
	Core radius: e.g. input --core_radius=0.5
	Outskirt radius: e.g. input --outskirt_radius=1.0
	Grid size: e.g. input --grid_size=1000

-h  Help file
--FeH_m=<arg>  Slope of the measured FeH gradient
--targ=<arg> Name of galaxy
--directory=<arg> Directory where galfit files are 
--subpix=<arg> Subpixels for 2D radii
--core_radius=<arg> Radius of the core in Re
--outskirt_radius=<arg> Radius of the outskirts in Re
--grid_size=<arg> Grid over which to forward model

"""

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs import WCS
import astropy.units as u
from astropy.convolution import Moffat2DKernel, convolve
from photutils.aperture import aperture_photometry, RectangularAperture, CircularAperture, EllipticalAperture, CircularAnnulus, RectangularAnnulus
from scipy import optimize
import math
import time
import numpy.ma as ma
from docopt import docopt
from scipy.integrate import quad

#Function flow
# Read in LEGA-C catalogue for image parameters
catalogue = pd.read_csv('intrinsic_model_cat.csv', header=0)
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

my_sample_df = LEGAC_final_df.merge(catalogue, on='SPECT_ID').drop_duplicates(subset='SPECT_ID')

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
    
def new_dimensions(n, m):
    """Return the new number of dimensions for a square grid when you divide n pixels into m subpixels.

    Parameters
    ----------
    n : int
        Original number of grid dimensions
    m : int
        Desired number of subpixels (must be an odd perfect square)

    Returns
    -------
    int(n * np.sqrt(m)) : int
        New number of grid dimensions
    """

    return int(n * np.sqrt(m))

def radii_2d(n, m):
    """Return the 2D radii (each pixel averaged over m subpixels).

    Parameters
    ----------
    n : int
        Original number of grid dimensions
    m : int
        Desired number of subpixels (must be an odd perfect square)

    Returns
    -------
    mean_grid : tuple
        New 2D radii in units of pixels
    """

    #Make a big grid made up of the number of desired subpixels
    big_grid = np.zeros((new_dimensions(n, m), new_dimensions(n, m)))

    #Calculate the distance between each subpixel and the centre of the grid
    for i in range(len(big_grid)):
        for j in range(len(big_grid[i])):
            big_grid[i][j] = np.sqrt(abs(i - len(big_grid) // 2)**2 + abs(j - len(big_grid) // 2)**2)      

    #Scale these distances down to the correct scale based on the number of subpixels
    sub_grid = big_grid/np.sqrt(m)

    #Reshape these distances to be in their correct subpixel groupings
    subpixels = []
    for i in range(0, len(sub_grid), int(np.sqrt(m))):
        subpix_row = []
        for j in range(0, len(sub_grid), int(np.sqrt(m))):
            subpix_row.append(sub_grid[i:i+int(np.sqrt(m)), j:j+int(np.sqrt(m))])
        subpixels.append(subpix_row)

    #Calculate the mean of the subpixels in each pixels and return to original sized grid
    mean_grid = np.zeros((int(n), int(n)))
    for i in range(len(subpixels)):
        for j in range(len(subpixels[i])):
            mean_grid[i][j] = np.mean(subpixels[i][j])

    return mean_grid
    
def intrinsic_FeH(targ, profile, m, subpix):
    """Return an intrinsic 2D nFe/nH gradient.

    Parameters
    ----------
    targ : str
        The name of the target
    profile : tuple
        The spectral profile
    m : float
        Slope to give intrinsic gradient
    subpix : int
        The number of radii subpixels 

    Returns
    -------
    intrinsic_grad_2d : tuple
        The intrinsic 2D nFe/nH gradient
    """

    #Get the integrated FeH abundance to use as an intercept
    integrated_FeH = catalogue['full Fe'][catalogue['SPECT_ID'] == targ]

    #Make 2D array of radii in units of pixels
    radii = radii_2d(int(len(profile)), subpix)

    #Give an intrinsic gradient 
    #intrinsic_grad_2d = 10**((m*np.log10(radii) + integrated_FeH) + 4.5) #units of pixels, 4.5 is subtracting the solar value so this is un-logged
    intrinsic_grad_2d = 10**((m*np.log10(radii)) + 4.5)

    return intrinsic_grad_2d

def Fe_profile(galfit_img, intrinsic_grad_2d):
    """Return the intrinsic 2D nFe profile.

    Parameters
    ----------
    galfit_img : tuple
        The galfit model = the intrinsic 2D nH profile
    intrinsic_grad_2d : tuple
        The intrinsic 2D nFe/nH gradient

    Returns
    -------
    galfit_img * intrinsic_grad_2d : tuple
        The intrinsic 2D nFe profile
    """

    return galfit_img * intrinsic_grad_2d

def generate_psf(gamma, alpha):
    """Return the PSF (2D Moffat by which the image will be convolved).

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
    moffat_psf = Moffat2DKernel(gamma, alpha)
    psf = moffat_psf.array

    return psf

def wavelength_solution(header, spec2d):
    """Return the wavelength solution for a given spectrum.  Note this is observed.

    Parameters
    ----------
    header : Header
        Fits header for corresponding 2D spectrum
    spec2d : tuple
        2D spectrum

    Returns
    -------
    pixels : tuple
        Pixel array
    wavelength : tuple
        Wavelength array
    """

    crpix1 = header['CRPIX1']
    crval1 = header['CRVAL1']
    cdelt = header['CD1_1']
    pixels = np.linspace(1, len(spec2d.T), len(spec2d.T)).astype(int)
    wavelength = np.arange(0, len(spec2d.T)) * cdelt + crval1
    return pixels, wavelength

def moffat(x, x0, amplitude, gamma, alpha):
    """From van Houdt+2021, same as in astropy Moffat1D.
    """
    return amplitude*(1 + ((x - x0)**2/gamma**2))**-alpha

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

def profile_mask(flux2D, wht2D, wave):
    """Return the 1D mask, 1D and 2D sky spectra, and the Gaussian profile for a 2D spectrum.
    Steps 1-3 of Aliza's code.
    
    Parameters
    ----------
    flux2D : tuple
        2D flux 
    wht2D : tuple
        2D weights (inverse variances)
    wave : tuple
        Wavelengths 
        
    Returns
    -------
    mask1D : tuple
        1D mask masking out bright skylines
    sky2D : tuple
        2D sky spectrum
    sky1D : tuple
        1D sky spectrum
    moffat_profile : dict
        Parameters of fitted Gaussian profile
    """
    
    ################################################
    ## Step 1: Make sky spectrum and skyline mask ##
    ################################################
    #Identify edge columns that == 0
    midrow = int(flux2D.shape[0]/2) #this is # of rows/2
    mask1D = flux2D[midrow] != 0
    
    #Make sky spectrum by collapsing over spatil axis
    #Remove 0's from weight spectrum
    num_wht = np.copy(wht2D)
    for i in range(len(num_wht)):
        for j in range(len(num_wht[i])):
            if num_wht[i][j] == 0.0:
                num_wht[i][j] = np.nan
                
    sky2D = 1/num_wht
    sky1D = np.nansum(sky2D, axis=0)
    
    #Mask out edges and use percentiles to find where we should mask lines
    skycut = np.nanpercentile(sky1D[mask1D][5:-5], 85)
    
    ##########################################
    ## Step 2: Make profile (mask skylines) ##
    ##########################################
    #Make a 2D mask that masks out the edges of the spectrum and where the sky is very bright
    mask2D = np.array([mask1D & (sky1D < skycut)]*flux2D.shape[0])
    profile = np.nansum(flux2D*mask2D, axis=1) #Profile is spatial, so you collapse over wavelength
    
    #########################################
    ## Step 3: Fit profile with a gaussian ##
    #########################################
    xfit = np.arange(0, len(profile), 1)
    yfit = profile*(profile > 0)
    popt, _ = optimize.curve_fit(moffat, xfit, yfit,p0=[np.nanargmax(profile), np.nanmax(profile), 1.5, 1.1])
    ctr = popt[0]
    ampl = popt[1]
    gamma = popt[2]
    alpha = popt[3]
    yfit = moffat(xfit, *popt)
    
    moffat_profile = {'ctr': ctr, 'ampl': ampl, 'gamma': gamma, 'alpha': alpha, 'xfit': xfit, 'yfit': yfit}
    return mask1D, sky2D, sky1D, moffat_profile, popt

def weights_1d(flux2D, wht2D, wave):
    """Return the core and outskirt optimally-extracted spectra, using fractional rows.
    Steps 4-5 of Aliza's code.

    Parameters
    ----------
    flux2D : tuple
        2D flux 
    wht2D : tuple
        2D weights (inverse variances)
    wave : tuple
        Wavelengths 

    Returns
    -------
    core_spectrum : DataFrame
        Core spectrum (wave, flux, noise)
    outskirt_spectrum : DataFrame
        Outskirt spectrum (wave, flux, noise)
    """

    ################
    ## Steps 1-3 ##
    ###############
    mask1D, sky2D, sky1D, moffat_profile, popt = (profile_mask(flux2D, wht2D, wave))
    
    ##############################
    ## Step 4: bin the spectra ##
    #############################
    percentile_35 = moffat_percentile(moffat_profile['yfit'], popt, 35)
    percentile_65 = moffat_percentile(moffat_profile['yfit'], popt, 65)
    percentile_3 = moffat_percentile(moffat_profile['yfit'], popt, 3)
    percentile_97 = moffat_percentile(moffat_profile['yfit'], popt, 97)
    Re = percentile_65 - percentile_35
    
    #Centre bin = 1Re
    centre_incl = (moffat_profile['xfit'] >= math.floor(moffat_profile['ctr'] - Re/2)) & (moffat_profile['xfit'] <= math.ceil(moffat_profile['ctr'] + Re/2))
    centre_yfit = np.copy(moffat_profile['yfit'])
    centre_yfit[math.floor(moffat_profile['ctr'] - Re/2)] = moffat_profile['yfit'][math.floor(moffat_profile['ctr'] - Re/2)]*(moffat_profile['ctr'] - Re/2 - math.floor(moffat_profile['ctr'] - Re/2))
    centre_yfit[math.ceil(moffat_profile['ctr'] + Re/2)] = moffat_profile['yfit'][math.ceil(moffat_profile['ctr'] + Re/2)]*(moffat_profile['ctr'] + Re/2 - math.floor(moffat_profile['ctr'] + Re/2))
    centre_yfit = centre_yfit/np.sum(centre_yfit[centre_incl])
    centre_yfit[~centre_incl] = 0.0

    #Outer bin
    #Get rid of flux in centre
    outer_excl = (moffat_profile['xfit'] >= moffat_profile['ctr'] - Re/2) & (moffat_profile['xfit'] <= moffat_profile['ctr'] + Re/2)
    outer_yfit = np.copy(moffat_profile['yfit'])
    outer_yfit[math.floor(moffat_profile['ctr'] - Re/2)] = moffat_profile['yfit'][math.floor(moffat_profile['ctr'] - Re/2)]*(1 - (moffat_profile['ctr'] - Re/2 - math.floor(moffat_profile['ctr'] - Re/2)))
    outer_yfit[math.ceil(moffat_profile['ctr'] + Re/2)] = moffat_profile['yfit'][math.ceil(moffat_profile['ctr'] + Re/2)]*(1 - (moffat_profile['ctr'] + Re/2 - math.floor(moffat_profile['ctr'] + Re/2)))
    outer_yfit[outer_excl] = 0.0
    
    #Get rid of flux on edges
    outer_yfit[math.floor(percentile_3)] = outer_yfit[math.floor(percentile_3)]*(percentile_3 - math.floor(percentile_3))
    outer_yfit[:math.floor(percentile_3)] = 0.0
    outer_yfit[math.ceil(percentile_97)] = outer_yfit[math.ceil(percentile_97)]*(percentile_97 - math.floor(percentile_97))
    outer_yfit[math.ceil(percentile_97)+1:] = 0.0
    
    outer_yfit = outer_yfit/np.sum(outer_yfit[~outer_excl])

    return centre_yfit, outer_yfit

def get_1d_weights(targ):
    targ_mask, targ_id = targ.split('_')[0], targ.split('_')[1]

    #Import 2D spectrum
    path = '/Users/chloecheng/Documents/LEGAC_resolved/2Dspectra/legac_' + targ_mask
    spec_file = fits.open(glob.glob(path + '*_spec2d_%s.fits' %targ_id)[0])
    header_spec = spec_file[0].header
    spec2d = spec_file[0].data
    spec_file.close()

    wht_file = fits.open(glob.glob(path + '*_wht2d_%s.fits' %targ_id)[0])
    wht2d = wht_file[0].data
    wht_file.close()

    wave = wavelength_solution(header_spec, spec2d)[1]

    core_weights, outskirt_weights = weights_1d(spec2d, wht2d, wave)
    return core_weights, outskirt_weights

def gradient_extraction(img_size, H_convolved, Fe_convolved, core_weights, outskirt_weights):
    """Return the model [Fe/H] in the core and outskirts.

    Parameters
    ----------
    img_size : int
        Length of the image in pixels
    H_convolved : tuple
        The convolved 2D nH profile
    Fe_convolved : tuple
        The convolved 2D nFe profile
    core_weights : tuple
        The weights in the core extraction region
    outskirt_weights : tuple
        The weights in the outskirt extraction region

    Returns
    -------
    m_conv : float
        Extracted slope
    b_conv : float
        Extracted intercept
    """

    #Define the slit
    slit = RectangularAperture((img_centre(img_size) - 1, img_centre(img_size) - 1), w=1/0.205, h=len(H_convolved), theta=Angle(0*u.deg))
    
    #Mask each convolved image with the slit
    slit_mask = slit.to_mask()
    H_masked = slit_mask.multiply(H_convolved)
    Fe_masked = slit_mask.multiply(Fe_convolved)
    
    #Collapse over the slit width
    H_collapsed = np.sum(H_masked, axis=1)
    Fe_collapsed = np.sum(Fe_masked, axis=1)
    
    #Apply the weights to the 1D profile and sum to get the final value
    core_H = np.sum(H_collapsed * core_weights)
    outskirt_H = np.sum(H_collapsed * outskirt_weights)
    core_Fe = np.sum(Fe_collapsed * core_weights)
    outskirt_Fe = np.sum(Fe_collapsed * outskirt_weights)

    #Calculate gradient in pixel space
    #m_conv = ((np.log10(outskirt_Fe/outskirt_H) - 4.5) - (np.log10(core_Fe/core_H) - 4.5))/(np.log10(outskirt_pix) - np.log10(core_pix))
    #b_conv = (np.log10(outskirt_Fe/outskirt_H) - 4.5) - m_conv*np.log10(outskirt_pix) #Not sure if I really need this
    convolved_core_FeH = np.log10(core_Fe/core_H) - 4.5 
    convolved_outskirt_FeH = np.log10(outskirt_Fe/outskirt_H) - 4.5

    return convolved_core_FeH, convolved_outskirt_FeH 
    
def intrinsic_model(targ, directory, m, subpix):
    """Return the slope and intercept of the extracted model gradient.
    This is the whole workflow of the model algorithm.  Can be run in a loop for a grid of input slopes m.

    Parameters
    ----------
    targ : str
        The name of the target
    directory : str
        Directory where the galfit models are
    m : float
        Input slope
    subpix : int
        Number of radius subpixels to average over

    Returns
    -------
    m_conv : float
        Extracted slope
    b_conv : float
        Extracted intercept
    """

    targ_df = my_sample_df[my_sample_df['SPECT_ID'] == targ]
    targ_gamma = catalogue['gamma'][catalogue['SPECT_ID'] == targ].values[0] #In pixels
    targ_alpha = catalogue['alpha'][catalogue['SPECT_ID'] == targ].values[0]

    #Generate 2D H profile
    profile = spectral_profile(targ)[0]
    img_size = len(profile)
    pixscale = 0.205
    galfit_input(targ, directory, img_size, img_centre(img_size), pixscale, targ + '_model', targ + '_model')
    galfit_model(directory, targ + '_model')

    galfit_file = fits.open('/Users/chloecheng/Documents/LEGAC_resolved/galfit_imgs/%s/%s_galfit.fits' %(directory, targ + '_model'))
    H_intrinsic = galfit_file[0].data
    galfit_file.close()

    #Generate 2D FeH profile
    FeH_intrinsic = intrinsic_FeH(targ, profile, m, subpix) #Units of pixels 

    #Generate 2D Fe profile
    Fe_intrinsic = Fe_profile(H_intrinsic, FeH_intrinsic)

    #Convolve 2D Fe and H profiles with seeing
    psf = generate_psf(targ_gamma, targ_alpha) #Units of pixels
    H_convolved = convolve(H_intrinsic, psf)
    Fe_convolved = convolve(Fe_intrinsic, psf)

    #Get weights
    core_weights, outskirt_weights = get_1d_weights(targ)

    #Extract convolved gradient
    convolved_core_FeH, convolved_outskirt_FeH = gradient_extraction(img_size, H_convolved, Fe_convolved, core_weights, outskirt_weights)
    return convolved_core_FeH, convolved_outskirt_FeH 
    
def best_fit_m(path, FeH_m, targ, directory, subpix, core_radius, outskirt_radius, grid_size):
    """Return the best-fitting intrinsic slope of the [Fe/H] gradient.
    
    Parameters
    ----------
    FeH_m : float
    	Slope of the measured [Fe/H] gradient
    targ : str
        Name of the target
    directory : str
        Directory to save files
    subpix : int
        Number of radius subpixels to average over
    core_radius : float
        Core radius value (pix or Re)
    outskirt_radius : float
        Outskirt radius value (pix or Re)
    """
    
    #Define grid of intrinsic slopes and intercepts
    m_grid = np.linspace(np.nanmin(catalogue['m'].to_list())-10, np.nanmax(catalogue['m'].to_list())+10, int(grid_size))
    
    #Calculate resulting slopes and intercepts in convolved space
    m_conv_grid = []
    b_conv_grid = []
    for i in range(len(m_grid)):
        grid_data = intrinsic_model(targ, directory, m_grid[i], int(subpix))
        m_conv_grid.append((grid_data[1] - grid_data[0])/(np.log10(float(outskirt_radius)) - np.log10(float(core_radius)))) #can be in pix or Re
        b_conv_grid.append(grid_data[1] - m_conv_grid[i]*np.log10(float(outskirt_radius))) #still not sure if I really need this
    
    #Find the difference between the real slope and each convolved model slope
    residuals = []
    for i in range(len(m_grid)):
        residuals.append(abs(float(FeH_m) - m_conv_grid[i]))
    
    #Write to file
    model_df = pd.DataFrame({'intrinsic': m_grid[np.argmin(residuals)], 'convolved': m_conv_grid[np.argmin(residuals)], 'residuals': residuals, 'measured': FeH_m})
    model_df.to_csv(path + '/intrinsic_model_files/%s_intmodel.dat' %targ, index=None, sep='\t')
    
    return m_grid[np.argmin(residuals)], m_conv_grid[np.argmin(residuals)], residuals
    
if __name__ == '__main__':
	arguments = docopt(__doc__)
	
	intrinsic, convolved, residuals = best_fit_m(arguments['--path'], arguments['--FeH_m'], arguments['--targ'], arguments['--directory'], arguments['--subpix'], arguments['--core_radius'], arguments['--outskirt_radius'], arguments['--grid_size'])