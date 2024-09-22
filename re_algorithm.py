import numpy as np
import pandas as pd
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
from scipy.signal import resample

font = {'family':'serif',
        'weight':'normal',
        'size':20
}
plt.rc('font',**font)

# Read in LEGA-C catalogue for image parameters
my_cat = pd.read_csv('zuvj_LEGAC_cat.csv', header=0, index_col=0)
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
    
    #Normalize the profile to add to 1
    norm_profile = profile/np.nansum(profile)
    norm_noise = noise_profile/np.nansum(profile)
    
    return norm_profile, norm_noise
    
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
        
def generate_psf(seeing, img_length, target):
    """Return the PSF (2D Gaussian by which the image will be convolved).
    
    Parameters
    ----------
    seeing : float
    	Width of the Gaussian (seeing by which to convolve)
    img_length : float
    	Size of the image (length of the spectral profile)
    target : str
    	Target name (mask + ID, i.e. 'M5_172669')
    	
    Returns
    -------
    psf : tuple
    	2D image array containing the Gaussian PSF
    """

    #Generate a 2D Gaussian PSF
    gaussian_psf = Gaussian2DKernel(seeing)
    
    #Pad with zeroes to get desired shape
    y, x = gaussian_psf.array.shape
    y_pad = (img_length - y)
    x_pad = (img_length - x)
    psf = np.pad(gaussian_psf.array, ((y_pad//2, y_pad//2 + y_pad%2), (x_pad//2, x_pad//2 + x_pad%2)), mode='constant')
    
    #Write to fits file
    hdu = fits.PrimaryHDU(psf)
    hdu.writeto('/galfit/psfs/psf_%s_%s.fits' %(target, str(seeing)), overwrite=True)
    
    return psf

def galfit_input(targ, psf_targ, directory, img_box, img_ctr, seeing, pixscale, output_name, input_name):
    """Write a galfit input file.
    
    Parameters
    ----------
    targ : str
    	Target name (mask + ID, i.e. 'M5_172669')
    psf_targ : str
    	Name of target in PSF image file
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
			'/Users/chloecheng/packages/galfit/psfs/psf_' + str(psf_targ) + '_' + str(seeing) +  '.fits #Input PSF image and diffusion kernel', 
			'none  #PSF fine sampling factor', 
			'none  #Bad pixel mask', 
			'none  #File with parameter constraints', 
			'1  ' + str(img_box) + '  1  ' + str(img_box) + '  #Image region to fix (xmin xmax ymin ymax)', 
			str(img_box+1) + '  ' + str(img_box+1) + '  #Size of the convolution box (x y)', 
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
    
def gaussian(x, amplitude, mean, stddev):
    """Return a Gaussian function.  To be used with curve_fit.
    
    Parameters
    ----------
    x : tuple
        x-data
    amplitude : float
        Amplitude of the Gaussian
    mean : float
        Mean of the Gaussian
    stddev : float
        Standard deviation of the Gaussian
    
    Returns
    -------
    amplitude * np.exp(-((x - mean) / 4 / stddev)**2) : tuple
        Gaussian function
    """
    
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

def image_profile(img_length, spec_arcsec, img_arcsec, pixscale, directory, galfit_model):
    """Return the normalized, resampled profile from the galfit model image.
    
    Parameters
    ----------
    img_length : float
        Size of the image (length of the spectrum profile)
    spec_arcsec : float
        Length of the spectrum in arcsec
    img_arcsec : float
        Length of the image in arcsec
    pixscale : float
        Pixelscale [arcsec/pixel]
    directory : str
        Directory where your output files are kept
    galfit_model : str
        Filename of desired galfit model
    
    Returns
    -------
    norm_profile_img : tuple
        Normalized profile from the galfit model image
    """
    
    # Define slit aperture and mask
    # for image centre, need to subtract 1 because of the 1 counting in galfit
    # aperture = RectangularAperture((img_centre(img_length) - 1, img_centre(img_length) - 1), w=1/pixscale, h=8/pixscale, theta=Angle(0*u.deg))
    # aperture_mask = aperture.to_mask()
    
    # Read in galfit image
    galfit_file = fits.open('/galfit_imgs/%s/%s_galfit.fits' %(directory, galfit_model))
    galfit_img = galfit_file[0].data
    galfit_file.close()
    
    # Make own aperture/mask because the photutils package sometimes adds or subtracts 1 pixel for some reason
    aperture_mask = np.zeros(galfit_img.shape)
    aperture_mask[:, int((img_centre(img_length) - 1 - 5)):int((img_centre(img_length) - 1 + 5))] = 1.0 # Define everywhere where the slit is
    
    # Mask the image
    # weighted_img = aperture_mask.multiply(galfit_img)
    weighted_img = galfit_img * aperture_mask
    
    # Collapse along slit width (basically summing over the centre 10 columns)
    profile_img = np.nansum(weighted_img, axis=1)
    
    # Normalize
    norm_profile_img = profile_img/np.nansum(profile_img)
    
    ### Resample ###
    spec_pixscale = np.arange(0, spec_arcsec, 0.205)
    img_pixscale = np.arange(0, img_arcsec, 0.1)
    high_pixscale = np.arange(0, img_arcsec, 0.005)
    
    # Interpolate onto pixelscale of 0.005''/pixel
    interp_profile_img = np.interp(high_pixscale, img_pixscale, norm_profile_img)
    
    # Fit with a Gaussian
    popt_profile_img, _ = curve_fit(gaussian, high_pixscale, interp_profile_img, p0=[np.max(interp_profile_img), np.mean(high_pixscale), 1])
    
    return spec_pixscale, high_pixscale, popt_profile_img
    
def max_abs_res(spec_profile, img_profile):
    """Return the maximum absolute residual between the two profiles.
    
    Parameters
    ----------
    spec_profile : tuple
        Profile from the spectrum
    img_profile : tuple
        Profile from the image
        
    Returns
    -------
    np.max(np.abs(spec_profile - img_profile)) : float
        Maximum absolute residual from the profile
    """

    return np.max(np.abs(spec_profile - img_profile))
    
def chisq_stat(flux, model, var):
    """Return the goodness of fit chi squared statistic.
    
    Parameters
    ----------
    flux : tuple
        Data flux
    model : tuple
        Model flux
    var : tuple
        Variance of the data
        
    Return
    ------
    np.nansum(chi) : float
        Reduced chisq statistic
    """
    
    chi = (flux - model)/var
    for i in range(len(chi)):
        if ~np.isfinite(chi[i]):
            chi[i] = np.nan
            
    return np.nansum(chi)
    
def single_run(seeing_start, profile_spectrum, noise_spec, target, psf_targ, directory, pixscale, output_name, input_name, stat_type):
    """Return the statistic and profile from a single calculation.
    
    Parameters
    ----------
    seeing_start : float
        Value of seeing to examine
    profile_spectrum : tuple
        Profile from the spectrum
    noise_spec : tuple
        Variance profile from the spectrum
    target : str
        Target name (mask + ID, i.e. 'M5_172669')
    psf_targ : str
    	Name of target in PSF image file
    directory : str
       Directory where your files are kept
    pixscale : float
        Pixelscale [arcsec/pixel]
    output_name : str
        Name of your galfit model file
    input_name : str
        Name of your galfit input file
    stat_type : str
       'residual' or 'chisq'
               
    Returns
    -------
    new_stat : float
        Residual or chisquare
    profile_itr : tuple
        Profile from the image
    """
    
    img_size = int(((len(profile_spectrum)*0.205) + 1)/0.1)
    spec_arcsec = len(profile_spectrum)*0.205
    img_arcsec = (len(profile_spectrum)*0.205) + 1
    
    # Generate the PSF
    psf_itr = generate_psf(seeing_start, img_size, target)
    
    # Generate the galfit model (convolved with the PSF already)
    galfit_input(target, psf_targ, directory, img_size, img_centre(img_size), seeing_start, pixscale, output_name, input_name)
    galfit_model(directory, input_name)
    
    # Get profile fit parameters
    spec_pixscale, high_pixscale, popt_profile_img = image_profile(len(profile_spectrum), spec_arcsec, img_arcsec, pixscale, directory, output_name)
    
    # Get array of x-offsets to test
    xoffs = np.linspace(popt_profile_img[1] - 0.5, popt_profile_img[1] + 0.5, 100)
    
    # Compare to spectrum
    offset_stat = []
    for i in range(len(xoffs)):
        if stat_type == 'residual':
            offset_stat.append(max_abs_res(profile_spectrum, resample(gaussian(high_pixscale, popt_profile_img[0], xoffs[i], popt_profile_img[2]), len(spec_pixscale))))
        elif stat_type == 'chisq':
            offset_stat.append(chisq(profile_spectrum, resample(gaussian(high_pixscale, popt_profile_img[0], xoffs[i], popt_profile_img[2]), len(spec_pixscale)), noise_spec**2))
    
    new_stat = np.min(offset_stat)
    profile_itr = resample(gaussian(high_pixscale, popt_profile_img[0], xoffs[np.argmin(offset_stat)], popt_profile_img[2]), len(spec_pixscale))
    
    return new_stat, profile_itr
    
def iterate_convolution(seeing_start, initial_stat, stat_type, target, psf_targ, directory, pixscale, output_name, input_name, seeing_decrement):
    """Return the best-fitting image profile and the best seeing.
    
    Parameters
    ----------
    seeing_start : float
        Value of seeing to examine
    initial_stat : float
        Initial value of statistic (large)
    stat_type : str
       'residual' or 'chisq'
    target : str
        Target name (mask + ID, i.e. 'M5_172669')
    psf_targ : str
        Name of target in PSF image file
    directory : str
       Directory where your files are kept
    pixscale : float
        Pixelscale [arcsec/pixel]
    output_name : str
        Name of your galfit model file
    input_name : str
        Name of your galfit input file
    seeing_decrement : float
        Step-size by which to decrease seeing each run
    
    Returns
    -------
    seeing_start : float
        Best seeing
    profile_spectrum : tuple
        Profile from the spectrum
    profile_itr : tuple
        Profile from the image
    """

    # Get spectrum profile
    profile_spectrum, noise_spec = spectral_profile(target)
    
    # Calculate 1 statistic to initialize
    new_stat, profile_itr = single_run(seeing_start, profile_spectrum, noise_spec, target, psf_targ, directory, spec_arcsec, img_arcsec, pixscale, output_name, input_name, stat_type)
    seeing_start -= seeing_decrement
    
    # Now iterate
    while new_stat <= initial_stat:
        initial_stat = new_stat
        new_stat, profile_itr = single_run(seeing_start, profile_spectrum, noise_spec, target, psf_targ, directory, spec_arcsec, img_arcsec, pixscale, output_name, input_name, stat_type)
        seeing_start -= seeing_decrement
    
    return seeing_start, profile_spectrum, profile_itr
    
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
    
def calculate_Re(target, psf_targ, best_seeing, directory, pixscale, output_name, input_name):
    """Return the convolved effective radius.
    
    Parameters
    ----------
    target : str
        Target name (mask + ID, i.e. 'M5_172669')
    psf_targ : str
        Name of target in PSF image file
    best_seeing : float
        Best seeing value
    directory : str
        Directory where your galfit images are
    pixscale : float
        Desired pixelscale
    output_name : str
        Galfit model file name
    input_name : str
        Galfit input file name
    
    Returns
    -------
    Re : float
        Convolved effective radius in pixels
    circle_radii : tuple
        Array of radii where cumulative fluxes were taken
    rescale(circle_sums) : tuple
        Cumulative fluxes, rescaled from 0 to 1 on the y-axis
    """
    
    # Get the profile from the spectrum
    profile_spectrum, noise_spec = spectral_profile(target)
    
    # Generate a higher-resolution and larger image and convolve by the best seeing
    img_size = int((((len(profile_spectrum)*0.205)+1)/0.1)*2) # Try doubling for now
    img_ctr = img_centre(img_size)
    big_psf = generate_psf(best_seeing, img_size, psf_targ) # Generate a PSF image of the same size
    galfit_input(target, psf_targ, directory, img_size, img_ctr, best_seeing, pixscale, output_name, input_name) 
    galfit_model(directory, input_name)
    
    # Read in this model
    galfit_file = fits.open('/galfit_imgs/%s/%s_galfit.fits' %(directory, output_name))
    galfit_img = galfit_file[0].data
    galfit_file.close()
    
    # Make circular apertures of increasing radius
    circle_radii = np.linspace(1, img_size, 1000)
    circle_apertures = []
    for i in range(len(circle_radii)):
        if img_size % 2 == 0:
            circle_apertures.append(CircularAperture((img_ctr - 2, img_ctr - 2), circle_radii[i])) # If side length is even, subtract 2 (I think this is still a counting thing)
        else:
            circle_apertures.append(CircularAperture((img_ctr - 1, img_ctr - 1), circle_radii[i]))
        
    # Perform aperture sums
    circle_sums = []
    for i in range(len(circle_apertures)):
        circle_sums.append(aperture_photometry(galfit_img, circle_apertures[i])['aperture_sum'].value[0])
        
    # Calculate Re
    Re = circle_radii[find_nearest(rescale(circle_sums), 0.5)[0]]
    
    return Re, circle_radii, rescale(circle_sums)