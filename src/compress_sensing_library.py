import numpy as np
import sys
sys.path.append("../")
from structured_random_features.src.models.weights import V1_weights

# Packages for dct, dwt and fitting data
from scipy import fftpack as fft
import pywt
from pywt import wavedecn
from sklearn.linear_model import Lasso
from pathlib import Path

# Packages for images
from PIL import Image, ImageOps

def fig_save_path(img_nm, method, observation, save_nm):
    save_nm = save_nm.replace(" ", "_")
    Path("../figures/{method}/{img_nm}/{observation}".format(
        method = method, img_nm = img_nm, observation = observation)).mkdir(parents=True, exist_ok = True)
    return "../figures/{method}/{img_nm}/{observation}/{save_nm}.png".format(
        method = method, img_nm = img_nm, observation = observation, save_nm = save_nm)

def data_save_path(img_nm, method, observation, save_nm): 
    save_nm = save_nm.replace(" ", "_")
    return "../result/{method}/{img_nm}/{observation}/{save_nm}.csv".format(
        method = method, img_nm = img_nm, observation = observation, save_nm = save_nm)

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
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data
    
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
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
    ''' Generate random pixel arrays with its indexes length of sample size.
    
    
    Parameters
    ----------
    img_arr : array_like
        (n, m) sized data array
    
    sample_size : int
        Number of sample data to be collected
    
    Returns
    ----------
    C3D : array like
        (sample_size, n, m) shape array that only has one index open that corresponds to y vector per each (n, m) shape array
    
    y : vector
        Actual value of randomly selected indices
    '''
    n, m = img_arr.shape
#     sample = np.floor(n * m * sample_size).astype(int)
    rand_index = np.random.randint(0, n * m, sample_size)
    y = img_arr.flatten()[rand_index].reshape(sample_size, 1)
    
    y = y * np.sqrt(n * m)
    C = np.eye(n * m)[rand_index, :] * np.sqrt(n * m)
    C3D = C.reshape(sample_size, n, m)
    return C3D, y

# Generate Gaussian Weights
def generate_gaussian_variables(img_arr, num_cell):
    ''' Generate 3 dimentional arrays. Creates arrays of randomly generated gaussian 2 dimentional arrays as a weight W
    
    Parameters
    ----------
    img_arr : array_like
        (n, m) sized data array
        
    num_cell : int
        Number of blobs that will be used to be determining which pixles to grab and use
        
    Returns
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of gaussian weighted data
    
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
    '''
    
    n, m = img_arr.shape
    W = np.random.randn(num_cell, n, m)
    y = generate_Y(W, img_arr)
    return W, y

# Error Calculation by Frosbian Norm
def error_calculation(img_arr, reconst):
    ''' Compute mean error per each data using frosbian norm
        
    Parameters
    ----------
    img_arr : array_like
        (n, m) shape array of actual dataset
    
    
    reconst : array_like
        (n, m) shape array of reconstructed array
    
    Returns
    ----------
    error : float
        Computed normalized error value per each pixel
    '''
    n, m = img_arr.shape
    error = np.linalg.norm(img_arr - reconst, 'fro') / np.sqrt(m * n)
    return error

# Reconstruction (Current Methods: Fourier Base Transform, Wavelet Transform)
def fourier_reconstruct(W, y, alpha, sample_sz, n, m, fit_intercept) :
    ''' Reconstruct signals through cosine transform
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data
        
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
        
    alpha : float
        Penalty for fitting data onto LASSO function to search for significant coefficents
    
    sample_sz : int
        Number of sample collected
    
    n : int
        Height of each data
    
    m : int
        Width of each data
    
    fit_intercept : bool
        default set to false to prevent LASSO function to calculate intercept for model
        
    Returns
    ----------
    theta : array_like
        (num_V1_weights/sample_size, n * m) shape. Data after discrete fourier transform applied 
    
    reconstruct : array_like
        (n, m) shape array. Reconstructed image pixel array
        
    s : vector
        (num_V1_weights/sample_size, 1) shape. Coefficient value generated from fitting data to LASSO. Contains significant values with most of vector zeroed out.
    '''
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
    ''' Reconstruct signals through wavelet transform
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of weighted data
        
    y : vector
        (num_V1_weights/sample_size, 1) shape. Dot product of W and image
        
    alpha : float
        Penalty for fitting data onto LASSO function to search for significant coefficents
    
    sample_sz : int
        Number of sample collected
    
    n : int
        Height of each data
    
    m : int
        Width of each data
    
    fit_intercept : bool
        default set to false to prevent LASSO function to calculate intercept for model
    
    dwt_type : String
        type of dwt method to be used
        ex)'haar', 'db1', 'db2', ...
        
    lv : int
        Generate level of signal frequencies when dwt is used
        
    Returns
    ----------
        theta : array_like
        (num_V1_weights/sample_size, n * m) shape. Data after discrete fourier transform applied 
    
    reconstruct : array_like
        (n, m) shape array. Reconstructed image pixel array
        
    s_unravel : vector
        (num_V1_weights/sample_size, 1) shape. Coefficient value generated from fitting data to LASSO. Contains significant values with most of vector zeroed out.
    '''
    
    dwt_sample = wavedecn(W[0], wavelet = dwt_type, level = lv, mode = 'zero')
    coeff, coeff_slices, coeff_shapes = pywt.ravel_coeffs(dwt_sample)
    theta = np.zeros((len(W), len(coeff)))
    theta[0, :] = coeff 

    # Loop the wavedecn to fill theta
    for i in range(sample_sz):
        theta_i = wavedecn(W[i], wavelet= dwt_type, level = lv, mode = 'zero')
        theta[i, :] = pywt.ravel_coeffs(theta_i)[0]

    mini = Lasso(alpha = alpha, fit_intercept = False)
    mini.fit(theta, y)

    s = mini.coef_

    s_unravel = pywt.unravel_coeffs(s, coeff_slices, coeff_shapes)
    reconstruct = pywt.waverecn(s_unravel, dwt_type, mode = 'zero')
    
    return theta, reconstruct, s_unravel

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
    
    method : String
        Currently supporting dct (descrete cosine transform) and dwt (descrete wavelet transform)
        Default set to dct
        
    lv : int
        Generate level of signal frequencies when dwt is used
        Default set to 4
        
    dwt_type : String
        type of dwt method to be used
        Default set to db2
        
    Returns
    ----------
    theta : array_like
        (num_V1_weights/sample_size, n * m) shape. Data after discrete fourier transform applied 
    
    reconstruct : array_like
        (n, m) shape array. Reconstructed image pixel array
        
    s : vector
        (num_V1_weights/sample_size, 1) shape. Coefficient value generated from fitting data to LASSO. Contains significant values with most of vector zeroed out.
    '''
    
    sample_sz, n, m = W.shape
      
    if alpha == None :
        alpha = 1 * 50 / sample_sz
        
    if fit_intercept:
        raise Exception("fit_intercept = True not implemented")
    
    if (method == 'dct') :
        theta, s, reconstruct = fourier_reconstruct(W, y, alpha, sample_sz, n, m, fit_intercept)
    elif (method == 'dwt') :
        theta, reconstruct, s = wavelet_reconstruct(W, y, alpha, sample_sz, n, m, fit_intercept, dwt_type, lv)

        # Reform the image using sparse vector s with inverse descrete cosine
        
    if fit_intercept:
        reform += mini.intercept_ # not sure this is right
    
    #return theta, reformed img, sparse vectors
    return theta, reconstruct, s

def filter_reconstruction(num_cell, img_arr, cell_size, sparse_freq, filter_dim = (30, 30), alpha = None, rand_weight = False) :
    ''' 
    Parameters
    ----------
    
    
    Returns
    ----------
    
    '''
    #alpha parameter is dependent on the number of cell if alpha is not specified
    if (alpha == None) :
        alpha = 1 * 50 / num_cell
    
    # Retrieve image dimension
    n, m = img_arr.shape
    
    # Create Filter
    filt = np.zeros(filter_dim)
    filt_n, filt_m = filter_dim
        
    # Fix the V1 weights if the random_weight parameter is set to be true
    if (rand_weight == True):
        W = V1_weights(num_cell, filter_dim, cell_size, sparse_freq) 

    # Preprocess image and add zeros so the cols and rows would fit to the filter for any size
    if n % filt_n != 0 :
        new_n = n + (filt_n - (n % filt_n))
    else :
        new_n = n
    if m % filt_m != 0 :
        new_m = m + (filt_m - (m % filt_m))
    else :
        new_m = m

    img_arr_aug = np.zeros((new_n, new_m))
    img_arr_aug[:n, :m] = img_arr

    print(img_arr_aug.shape)
    i = 1 # counter
    result = np.zeros(img_arr.shape)
    cur_n, cur_m = (0, 0)
    num_work = (new_n * new_m) // (filt_n * filt_m)
    
    for pt in range(num_work):
#         if (i % (num_work // 5) == 0) :
#             print("iteration", i)
        # Randomize V1 weights for each batch if random_weight param is set to false
        if (rand_weight != True) :
            W = V1_weights(num_cell, filter_dim, cell_size, sparse_freq) 

        # keep track over height of the batches
        if (cur_m >= new_m) :
            cur_n += filt_n
            cur_m = 0

        nxt_m = cur_m + filt_m
        pt = img_arr_aug[cur_n : (cur_n + filt_n), cur_m : nxt_m]

        y = generate_Y(W, pt)
        W_model = W.reshape(num_cell, filt_n, filt_m)
        theta, reform, s = reconstruct(W_model, y, alpha)

        img_arr_aug[cur_n : (cur_n + filt_n), cur_m : nxt_m] = reform
        cur_m = nxt_m

        i+=1

    result = img_arr_aug[:n,:m]
    return result
