import numpy as np
import pandas as pd
import glob
from scipy import optimize
import math
from specutils.analysis import centroid
from specutils import Spectrum1D, SpectralRegion
from read_alf import Alf
from astropy.convolution import interpolate_replace_nans, Gaussian1DKernel, convolve, convolve_fft
import astropy.io.fits
import astropy.units as u

#Continuum-normalization functions (also normalizes noise)
def data_ranges(lambda_min, lambda_max, flux, wavelength, noise):
    """Return the data within the desired wavelength range.
    
    Parameters
    ----------
    lambda_min : float
        The lower bound of the wavelength range 
    lambda_max : float
        The upper bound of the wavelength range
    flux : tuple
        The flux data
    wavelength : tuple
        The wavelength data
    noise : tuple
        The noise data
        
    Returns
    -------
    wavelength_inrange : tuple
        The wavelength data within the desired wavelength range
    flux_inrange : tuple
        The flux data within the desired wavelength range
    noise_inrange : tuple
        The noise data within the desired wavelength range
    """
    
    rng = (wavelength >= lambda_min) & (wavelength <= lambda_max)
    wavelength_inrange = wavelength[rng]
    flux_inrange = flux[rng]
    noise_inrange = noise[rng]
    return wavelength_inrange, flux_inrange, noise_inrange

def poly_order(lambda_min, lambda_max):
    """Return the order of the polynomial by which to continuum-normalize the spectrum.
    
    Parameters
    ----------
    lambda_min : float
        The lower bound of the wavelength range
    lambda_max : float
        The upper bound of the wavelength range
        
    Returns
    -------
    (lambda_min/lambda_max)/100 : int
        The order of the polynomial by which to continuum-normalize the spectrum
    """
    
    return int((lambda_max - lambda_min)/100)

def continuum_normalize(lambda_min, lambda_max, flux, wavelength, noise):
    """Return the spectrum normalized by the fitted polynomial shape of the continuum.
    
    Parameters
    ----------
    lambda_min : float
        The lower bound of the wavelength range
    lambda_max : float
        The upper bound of the wavelength range
    flux : tuple
        The flux data
    wavelength: tuple
        The wavelength data
    noise : tuple
        The noise data
        
    Returns
    -------
    wavelength_inrange : tuple
        The wavelength data within the desired wavelength range
    flux_norm : tuple
        The continuum-normalized flux within the desired wavelength range
    noise_norm : tuple
        The continuum-normalized noise within the desired wavelength range
    """
    
    #Get the data in the desired wavelength range
    wavelength_inrange, flux_inrange, noise_inrange = data_ranges(lambda_min, lambda_max, flux, wavelength, noise)
    
    #Get the order of the polynomial 
    n = poly_order(lambda_min, lambda_max)
    
    #Fit an nth-order polynomial to the spectrum
    xrange = np.linspace(np.min(lambda_min), np.max(lambda_max), len(wavelength_inrange))
    polynomial_fit, cov = np.polyfit(xrange, flux_inrange, n, cov=True)
    poly_obj = np.poly1d(polynomial_fit)
    
    polynomial_fit_noise = np.polyfit(xrange, noise_inrange, n)
    poly_obj_noise = np.poly1d(polynomial_fit_noise)
    
    #Divide out the fitted polynomial to get a continuum-normalized spectrum
    flux_norm = flux_inrange/poly_obj(xrange)
    noise_norm = noise_inrange/poly_obj(xrange)
    #noise_norm = flux_norm*np.sqrt((noise_inrange/flux_inrange)**2 + (np.sqrt(np.diag(cov))/poly_obj(xrange))**2)
    return wavelength_inrange, flux_norm, noise_norm
    
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
    
def gaussian_extract(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

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
    gaussian_profile : dict
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
    popt, _ = optimize.curve_fit(gaussian_extract, xfit, yfit,p0=[np.nanmax(profile), np.nanargmax(profile), 1])
    ampl = popt[0]
    mu = popt[1]
    sigma = popt[2]
    yfit = gaussian_extract(xfit, *popt)
    
    gaussian_profile = {'ampl': ampl, 'mu': mu, 'sigma': sigma, 'xfit': xfit, 'yfit': yfit}
    return mask1D, sky2D, sky1D, gaussian_profile

def optimal_extraction_full(flux2D, wht2D, wave):
    """Return the full, optimally-extracted spectrum.
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
    spectrum : DataFrame
        Spectrum (wave, flux, noise)
    """
    
    ################
    ## Steps 1-3 ##
    ###############
    mask1D, sky2D, sky1D, gaussian_profile = (profile_mask(flux2D, wht2D, wave))
    
    ##################################################################
    ## Step 4: only include rows with flux > 0.1*gaussian amplitude ##
    ##################################################################
    incl = gaussian_profile['yfit'] > 0.1*gaussian_profile['ampl']
    yfit = gaussian_profile['yfit'] / np.sum(gaussian_profile['yfit'][incl])
    yfit[~incl] = 0.0
    
    ###################################################
    ## Step 5: Extract 1D science and noise spectrum ##
    ###################################################
    #This is the full spectrum
    mask2D = np.array([mask1D]*flux2D.shape[0])
    yfit2D = np.array([yfit]*flux2D.shape[1]).T
    weight = mask2D*yfit2D/sky2D

    #not sure where this comes from.. took it from mariskas code below
    spec = np.nansum(flux2D*weight, axis=0)/np.nansum(weight*yfit2D, axis=0)
    noise = np.sqrt(np.abs(1/np.nansum(weight*yfit2D, axis=0))) 
    spectrum = pd.DataFrame(np.array((wave, spec, noise, np.nansum(weight, axis=0), sky1D)).T, columns=['wave', 'flux', 'noise', 'weight', 'sky'])
    return spectrum

def optimal_extraction_int(flux2D, wht2D, wave):
    """Return the core and outskirt optimally-extracted spectra, using integer rows.
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
    mask1D, sky2D, sky1D, gaussian_profile = (profile_mask(flux2D, wht2D, wave))
    
    ##############################
    ## Step 4: bin the spectra ##
    #############################
    Re = 2*np.sqrt(2*np.log(2))*np.abs(gaussian_profile['sigma']) #FWHM = Re - this will be in units of pixels

    #Centre bin = 1Re
    centre_incl = (gaussian_profile['xfit'] >= (gaussian_profile['mu'] - np.round(Re/2, 0))) & (gaussian_profile['xfit'] <= (gaussian_profile['mu'] + np.round(Re/2, 0)))
    centre_yfit = gaussian_profile['yfit']/np.sum(gaussian_profile['yfit'][centre_incl])
    centre_yfit[~centre_incl] = 0.0

    #Outer bin 
    #Get rid of flux on very edges and also in the centre
    outer_incl = gaussian_profile['yfit']>0.1*gaussian_profile['ampl']
    outer_yfit = gaussian_profile['yfit']/np.sum(gaussian_profile['yfit'][outer_incl & ~centre_incl])
    outer_yfit[~outer_incl] = 0.0
    outer_yfit[centre_incl] = 0.0
    
    ###################################################
    ## Step 5: Extract 1D science and noise spectrum ##
    ###################################################
    #This is the full spectrum
    mask2D = np.array([mask1D]*flux2D.shape[0])
    centre_yfit2D = np.array([centre_yfit]*flux2D.shape[1]).T
    outer_yfit2D = np.array([outer_yfit]*flux2D.shape[1]).T

    centre_weight = mask2D*centre_yfit2D/sky2D
    outer_weight = mask2D*outer_yfit2D/sky2D

    #not sure where this comes from.. took it from mariskas code below
    centre_spec = np.nansum(flux2D*centre_weight,axis=0)/np.nansum(centre_weight*centre_yfit2D,axis=0)
    outer_spec = np.nansum(flux2D*outer_weight,axis=0)/np.nansum(outer_weight*outer_yfit2D,axis=0)
    centre_noise = np.sqrt(np.abs(1/np.nansum(centre_weight*centre_yfit2D,axis=0))) 
    outer_noise = np.sqrt(np.abs(1/np.nansum(outer_weight*outer_yfit2D,axis=0)))
    
    core_spectrum = pd.DataFrame(np.array((wave, centre_spec, centre_noise, np.nansum(centre_weight, axis=0), sky1D)).T, columns=['wave', 'flux', 'noise', 'weight', 'sky'])
    outskirt_spectrum = pd.DataFrame(np.array((wave, outer_spec, outer_noise, np.nansum(outer_weight, axis=0), sky1D)).T, columns=['wave', 'flux', 'noise', 'weight', 'sky'])
    return core_spectrum, outskirt_spectrum


def optimal_extraction_frac(flux2D, wht2D, wave):
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
    mask1D, sky2D, sky1D, gaussian_profile = (profile_mask(flux2D, wht2D, wave))
    
    ##############################
    ## Step 4: bin the spectra ##
    #############################
    Re = 2*np.sqrt(2*np.log(2))*np.abs(gaussian_profile['sigma']) #FWHM = Re - this will be in units of pixels
    
    #Centre bin = 1Re
    centre_incl = (gaussian_profile['xfit'] >= math.floor(gaussian_profile['mu'] - Re/2)) & (gaussian_profile['xfit'] <= math.ceil(gaussian_profile['mu'] + Re/2))
    centre_yfit = np.copy(gaussian_profile['yfit'])
    centre_yfit[math.floor(gaussian_profile['mu'] - Re/2)] = gaussian_profile['yfit'][math.floor(gaussian_profile['mu'] - Re/2)]*(gaussian_profile['mu'] - Re/2 - math.floor(gaussian_profile['mu'] - Re/2))
    centre_yfit[math.ceil(gaussian_profile['mu'] + Re/2)] = gaussian_profile['yfit'][math.ceil(gaussian_profile['mu'] + Re/2)]*(gaussian_profile['mu'] + Re/2 - math.floor(gaussian_profile['mu'] + Re/2))
    centre_yfit = centre_yfit/np.sum(centre_yfit[centre_incl])
    centre_yfit[~centre_incl] = 0.0

    #Outer bin
    #Get rid of flux on very edges and also in the centre
    outer_incl = gaussian_profile['yfit'] > 0.1 * gaussian_profile['ampl']
    outer_excl = (gaussian_profile['xfit'] >= gaussian_profile['mu'] - Re/2) & (gaussian_profile['xfit'] <= gaussian_profile['mu'] + Re/2)
    outer_yfit = np.copy(gaussian_profile['yfit'])
    outer_yfit[math.floor(gaussian_profile['mu'] - Re/2)] = gaussian_profile['yfit'][math.floor(gaussian_profile['mu'] - Re/2)]*(1 - (gaussian_profile['mu'] - Re/2 - math.floor(gaussian_profile['mu'] - Re/2)))
    outer_yfit[math.ceil(gaussian_profile['mu'] + Re/2)] = gaussian_profile['yfit'][math.ceil(gaussian_profile['mu'] + Re/2)]*(1 - (gaussian_profile['mu'] + Re/2 - math.floor(gaussian_profile['mu'] + Re/2)))
    outer_yfit = outer_yfit/np.sum(outer_yfit[outer_incl & ~outer_excl])
    outer_yfit[~outer_incl] = 0.0
    outer_yfit[outer_excl] = 0.0
    
    ###################################################
    ## Step 5: Extract 1D science and noise spectrum ##
    ###################################################
    #This is the full spectrum
    mask2D = np.array([mask1D]*flux2D.shape[0])
    centre_yfit2D = np.array([centre_yfit]*flux2D.shape[1]).T
    outer_yfit2D = np.array([outer_yfit]*flux2D.shape[1]).T

    centre_weight = mask2D*centre_yfit2D/sky2D
    outer_weight = mask2D*outer_yfit2D/sky2D

    #not sure where this comes from.. took it from mariskas code below
    centre_spec = np.nansum(flux2D*centre_weight,axis=0)/np.nansum(centre_weight*centre_yfit2D,axis=0)
    outer_spec = np.nansum(flux2D*outer_weight,axis=0)/np.nansum(outer_weight*outer_yfit2D,axis=0)
    centre_noise = np.sqrt(np.abs(1/np.nansum(centre_weight*centre_yfit2D,axis=0))) 
    outer_noise = np.sqrt(np.abs(1/np.nansum(outer_weight*outer_yfit2D,axis=0)))
    
    core_spectrum = pd.DataFrame(np.array((wave, centre_spec, centre_noise, np.nansum(centre_weight, axis=0), sky1D)).T, columns=['wave', 'flux', 'noise', 'weight', 'sky'])
    outskirt_spectrum = pd.DataFrame(np.array((wave, outer_spec, outer_noise, np.nansum(outer_weight, axis=0), sky1D)).T, columns=['wave', 'flux', 'noise', 'weight', 'sky'])
    return core_spectrum, outskirt_spectrum
    
def gaussian(x, a, x0, sigma, d):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + d

def FWHM(line, flux, wave, noise):
    """Return the normalized wavelengths & fluxes, the optimal Gaussian fit parameters, the FWHM, and the centroid of a line.
    
    Parameters
    ----------
    line : tuple
        Beginning and ending wavelengths of line
    flux : tuple
        Flux array
    wave : tuple
        Wavelength array
    noise : tuple
        Noise array
    
    Returns
    -------
    line_wave : tuple
        Normalized wavelength around line region
    line_flux : tuple
        Normalized flux around line region
    popt_line : tuple
        Best-fit Gaussian parameters
    FWHM_line : float
        FWHM of the line
    measured_centroid.value : float
        The centroid of the line
    """
    
    #Want to keep all of the same lines, but some spectra are missing lines at the edges due to different wavelength ranges
    if np.all(np.isnan(flux[(wave >= line[0]) & (wave <= line[1])])) or np.all(flux[(wave >= line[0]) & (wave <= line[1])] == 0.0):
        return None, None, None, None, None
#         line_wave = None
#         line_flux = None
#         popt_line = None
#         FWHM_line = None
#         measured_centroid = None
    else:
        #Measure FWHM
        line_wave, line_flux, line_noise = continuum_normalize(line[0], line[-1], flux, wave, noise)
        n_line = len(line_wave)
        mean_line = np.mean(line_wave)
        std_line = np.std(line_wave)
        try:
            popt_line, pcov_line = optimize.curve_fit(gaussian, line_wave, line_flux, p0=[np.max(line_flux), mean_line, std_line, 0])
            FWHM_line = 2*np.sqrt(2*np.log(2))*np.abs(popt_line[2])
        except:
            FWHM_line = np.nan

        #Find corresponding centroid
        spectrum = Spectrum1D(spectral_axis=line_wave*u.Angstrom, flux=line_flux.to_numpy()*u.ct)
        region = SpectralRegion(line[0]*u.Angstrom, line[-1]*u.Angstrom)
        measured_centroid = centroid(spectrum, region).value
    return line_wave, line_flux, popt_line, FWHM_line, measured_centroid

def spectral_resolution(lines, flux, wave):
    #Calculate the FWHM of each line
    FWHM_data = []
    FWHMs = []
    centroids = []
    for i in range(len(lines)):
        FWHM_data.append(FWHM(lines[i], flux, wave, np.zeros_like(wave)))
        FWHMs.append(FWHM_data[i][3])
        centroids.append(FWHM_data[i][4])
    
    #Interpolate over whole wavelength range
    kernel = Gaussian1DKernel(stddev = 1)
    FWHMs[:] = (value for value in FWHMs if value != None)
    centroids[:] = (value for value in centroids if value != None)
    interp_res = np.interp(wave, centroids, FWHMs)
    #interp_res_nonan = interpolate_replace_nans(interp_res, kernel, convolve = convolve_fft)
    return interp_res

def resolution(R, wave):
    """Return the spectral resolution in km/s.
    
    Parameters
    ----------
    R : float
        Spectral resolution (resolving power)
    wave : tuple
        Wavelength array
        
    Returns
    -------
    res_kms : tuple
        Resolution in km/s (allows for wavelength dependence)
    """
    
    c = 299792.458 #speed of light
    res_kms = R/wave*c #Convert to km/s
    return res_kms