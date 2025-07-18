# -*- coding: utf-8 -*-
""" Implements python port of gsolve.m """

# imports
import numpy as np
import math
from scipy.interpolate import griddata
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import median_filter

# implementation
def gsolve(Z: np.ndarray, B: np.ndarray, l: int, w) -> (np.ndarray, np.ndarray):
    '''
    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system’s response function g as well as the log film irradiance
    values for the observed pixels.

    Arguments:
        Z: N x P array for P pixels in N images
        B: is the log delta t, or log shutter speed, for image j
        l: lambda, the constant that determines smoothness
        w: is the weighting function value for pixel value
    Returns:
        g: solved g value per intensity, of shape 256 x 1
        le: log irradiance for sample pixels of shape P x 1
    '''

    N, P = Z.shape

    n = 256
    A = scipy.sparse.lil_matrix(((N * P) + n + 1, n + P), dtype='double') # init lil
    b = np.zeros((A.shape[0], 1), dtype='double')

    k = 0
    # for each pixel
    for i in range(N):
        # for each image
        for j in range(P):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + j] = -wij
            b[k, 0] = wij * B[i]
            k += 1

    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1
    k += 1

    # Include the smoothness equation
    for i in range(n - 2):
        A[k, i] = l 
        A[k, i + 1] = -2 * l 
        A[k, i + 2] = l 
        k += 1
    
    x = scipy.sparse.linalg.lsqr(A.tocsr(), b); 

    g = x[0][:n]
    lE = x[0][n:]

    return g, lE

def get_equirectangular_image(reflection_vector, hdr_image):
    '''
    Given a set of Reflection Vectors for all the pixels in the image
    along with the HDR image saved from the previous part, this function 
    returns the equirectangular image for the environment map that can be
    directly used in Blender for the next part.
    
    Arguments:
        reflection_vector: H x W x 3 array containing the reflection vector at each pixel across the three dimensions
        hdr_image: the LDR merged image from the previous part
    
    Returns:
        equirectangular_image: This is the equirectangular environment map that is to be used in the next part.
    '''

    H, W, C = hdr_image.shape
    rv_x, rv_y, rv_z = np.split(reflection_vector, 3, axis=2)

    # Step 1: Convert reflection vectors to spherical coordinates
    theta_ball = np.pi - np.arccos(np.clip(rv_y, -1, 1))
    phi_ball = np.arctan2(rv_z, rv_x)
    phi_ball[np.isnan(phi_ball)] = 0
    phi_ball += 3 * np.pi / 2
    phi_ball %= 2 * np.pi

    # Step 2: Create equirectangular grid
    EH, EW = 360, 720
    phi_1st_half = np.arange(np.pi, 2 * np.pi, np.pi / (EW // 2))
    phi_2nd_half = np.arange(0, np.pi, np.pi / (EW // 2))
    phi_ranges = np.concatenate((phi_1st_half, phi_2nd_half))
    theta_range = np.arange(0, np.pi, np.pi / EH)
    phis, thetas = np.meshgrid(phi_ranges, theta_range)

    # Step 3: Flatten coords and HDR values
    spherical_coord = np.concatenate((phi_ball, theta_ball), axis=2).reshape(-1, 2)
    spherical_vals = hdr_image.reshape(-1, 3)
    equirectangular_coord = np.stack((phis, thetas), axis=2).reshape(-1, 2)

    # Step 4: Mask invalid points
    valid_mask = ~np.isnan(spherical_coord).any(axis=1)
    valid_mask &= ~(np.linalg.norm(spherical_coord, axis=1) == 0)
    spherical_coord_valid = spherical_coord[valid_mask]
    spherical_vals_valid = spherical_vals[valid_mask]

    # Step 5: Interpolate in log space for each channel
    equirectangular_intensities = []
    for c in range(C):
        # Convert to log domain (add epsilon to avoid log(0))
        log_vals = np.log(spherical_vals_valid[:, c] + 1e-6)

        # Interpolate log intensities
        interp_log = griddata(
            spherical_coord_valid,
            log_vals,
            equirectangular_coord,
            method='linear',
            fill_value=np.nan
        )

        # Fallback to nearest for missing
        if np.any(np.isnan(interp_log)):
            nearest_log = griddata(
                spherical_coord_valid,
                log_vals,
                equirectangular_coord,
                method='nearest'
            )
            interp_log = np.where(np.isnan(interp_log), nearest_log, interp_log)

        # Convert back from log domain
        interp_exp = np.exp(interp_log)

        # Optional: smooth harsh noise in bright regions
        interp_exp = median_filter(interp_exp.reshape(EH, EW), size=3)

        equirectangular_intensities.append(interp_exp)

    # Step 6: Stack and fix any remaining NaNs
    equirectangular_image = np.stack(equirectangular_intensities, axis=2)
    nan_mask = np.isnan(equirectangular_image)
    if np.any(nan_mask):
        mean_val = np.nanmean(equirectangular_image)
        equirectangular_image[nan_mask] = mean_val


    # Apply median filtering across each channel to remove residual speckles
    for c in range(3):
        equirectangular_image[:, :, c] = median_filter(equirectangular_image[:, :, c], size=3)

    return equirectangular_image.astype(np.float32)
