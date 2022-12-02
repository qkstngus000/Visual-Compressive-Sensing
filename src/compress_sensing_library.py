import numpy as np
import sys
sys.path.append("../")
from structured_random_features.src.models.weights import V1_weights

# Packages for dct, dwt and fitting data
from scipy import fftpack as fft
import pywt
from pywt import wavedecn
from sklearn.linear_model import Lasso

# Packages for images
from PIL import Image, ImageOps

# Generate General Variables
def generate_Y(W, img):
    ''' Generate sample y vector variable for data reconstruction using constant matrix W (containing open indices). Function does inner product W matrix with image array to find sample y vector, 
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights/sample_size, n*m) shape array. Lists of weighted data
        
    img : array_like
          (n, m) shape image containing array of pixels
    
    Returns
    ----------
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
    
    '''
    
    num_cell = W.shape[0]
    n, m = img.shape
    W = W.reshape(num_cell, n*m)
    y = W @ img.reshape(n * m, 1)
    return y

def generate_V1_variables(num_cell, cell_size, sparse_freq, img):
    ''' Automatically generates variables needed for data reconstruction using V1 weights.
    
    Parameters
    ----------
    num_cell : int
        Number of blobs that will be used to be determining which pixles to grab and use
    
    cell_size : int
        Determines field size of opened and closed blob of data. Affect the data training
        
    sparse_freq : int
        Determines filed frequency on how frequently opened and closed area would appear. Affect the data training
    
    img : array_like
          (n, m) shape image containing array of pixels
    
    Returns
    ----------
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
    
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data
    
    '''
    # Get size of image
    dim = np.asanyarray(img).shape
    n, m = dim
    # Store generated V1 cells in W
    W = V1_weights(num_cell, dim, cell_size, sparse_freq) 
    
    # Retrieve y from W @ imgArr
    y = W @ img.reshape(n*m, 1)

    # Resize W to shape (num_cell, height of image, width of image) for fetching into function
    W = W.reshape(num_cell, dim[0], dim[1])
    return W, y

# Generate Classical Variables
def generate_classical_variables(img_arr, sample_size) :
    n, m = img_arr.shape
    sample = np.floor(n * m * sample_size).astype(int)
    rand_index = np.random.randint(0, n * m, sample)
    y = img_arr.flatten()[rand_index].reshape(sample, 1)
    
    classical_y = classical_y * np.sqrt(cn * cm)
    C = np.eye(cn * cm)[rand_index, :] * np.sqrt(cn * cm)
    C3D = C.reshape(classical_samp, cn, cm)
    return C3D, y

# Generate Gaussian Weights
def gaussian_W(num_cell, img_dim):
    n, m = img_dim
    W = np.random.randn(num_cell, n, m)
    return W

# Error Calculation by Frosbian Norm
def error_calculation(img_arr, reconst):
    n, m = img_arr.shape
    error = np.linalg.norm(img_arr - reconst, 'fro') / np.sqrt(cm*cn)
    return error

# Reconstruction (Current Methods: Fourier Base Transform, Wavelet Transform)
def fourier_reconstruct(W, y, alpha, sample_sz, n, m, fit_intercept) :
    theta = fft.dctn(W.reshape(sample_sz, n, m), norm = 'ortho', axes = [1, 2])
    theta = theta.reshape(sample_sz, n * m)

    ## Initialize Lasso and Fit data
    mini = Lasso(alpha = alpha, fit_intercept = fit_intercept)
    mini.fit(theta, y)

    ## Retrieve sparse vector s
    s = mini.coef_
    reconstruct = fft.idctn(s.reshape(n, m), norm='ortho', axes=[0,1])
    return theta, s, reconstruct

def wavelet_reconstruct(W, y, alpha, sample_sz, n, m, fit_intercept, dwt_type, lv) :
    dwt_sample = wavedecn(W[0], wavelet = dwt_type, level = lv)
    coeff, coeff_slices, coeff_shapes = pywt.ravel_coeffs(dwt_sample)
    theta = np.zeros((len(W), len(coeff)))
    theta[0, :] = coeff 

    # Loop the wavedecn to fill theta
    for i in range(samp):
        theta_i = wavedecn(W[i], wavelet= dwt_type, level = lv)
        theta[i, :] = pywt.ravel_coeffs(theta_i)[0]

    mini = Lasso(alpha = alpha, fit_intercept = False)
    mini.fit(theta, y)

    s = mini.coef_

    s_unravel = pywt.unravel_coeffs(s, coeff_slices, coeff_shapes)
    reconstruct = pywt.waverecn(s_unravel, w)
    
    return theta, s_unravel, reconstruct

def reconstruct(W, y, alpha = None, fit_intercept = False, method = 'dct', lv = 4, dwt_type = 'db2'):
    ''' Reconstruct gray-scaled image using sample data fitting into LASSO model
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data
        
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
        
    alpha : float
        Penalty for fitting data onto LASSO function to search for significant coefficents
    
    fit_intercept : bool
        default set to false to prevent LASSO function to calculate intercept for model
    
    Returns
    ----------
    theta : array_like
        (num_V1_weights/sample_size, n * m) shape. Data after discrete fourier transform applied 
    
    reformed : array_like
        (n, m) shape array. Reconstructed image pixel array
        
    s : vector
        (num_V1_weights/sample_size, 1) shape. Coefficient value generated from fitting data to LASSO. Contains significant values with most of vector zeroed out.
    '''
    
    sample_sz, n, m = W.shape
      
    if alpha == None :
        alpha = 1 * 50 / num_cell
        
    if fit_intercept:
        raise Exception("fit_intercept = True not implemented")
    
    if (method == 'dct') :
        theta, s, reconstruct = fourier_reconstruct(W, y, sample_sz, n, m, fit_intercept)
    elif (method == 'dwt') :
        theta, s, reconstruct = wavelet_reconstruct(W, y , sample_sz, n, m, fit_intercept, dwt_type, lv)

        # Reform the image using sparse vector s with inverse descrete cosine
        
    if fit_intercept:
        reform += mini.intercept_ # not sure this is right
    
    #return theta, reformed img, sparse vectors
    return theta, reconstruct, s