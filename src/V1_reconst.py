import numpy as np
import sys
sys.path.append("../")
from structured_random_features.src.models.weights import V1_weights

# Packages for fft and fitting data
from scipy import fftpack as fft
from sklearn.linear_model import Lasso

def generate_Y(W, img):
    ''' Fetch image pixel values to V1 weights to generate data for fitting comoress sensing
    
    Parameters
    ----------
    W : array_like
        (num_V1_weights, n*m) shape array. Lists of V1 weights.
        
    img : array_like
          (n, m) shape image array
    
    Returns
    ----------
    y : vector
        Dot product of W and image
    
    '''
    num_cell = W.shape[0]
    n, m = img.shape
    W = W.reshape(num_cell, n*m)
    y = W @ img.reshape(n * m, 1)
    return y

def generate_V1_variables(num_cell, cell_size, sparse_freq, img):
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


def reconstruct(W, y, alpha = None, fit_intercept = False,):
    # Function: reconstruct
    # Parameters:
    ##     W: An opened index for measurement
    ##     y: the value of the opened index W
    ##     alpha: panelty value to fit for Lasso
    ##     dim (n, m): image size that needs to be reformed

    # Return:
    ##     theta: matrix of W when FFT took in place
    ##     reformed: Reformed image in array
    ##     s: sparse vector s which is a estimated coefficient generated from LASSO
    sample_sz, n, m = W.shape
    
    
    if alpha == None :
        alpha = 1 * 50 / num_cell
        
    if fit_intercept:
        raise Exception("fit_intercept = True not implemented")
    
    ## WΨ
    theta = fft.dctn(W.reshape(sample_sz, n, m), norm = 'ortho', axes = [1, 2])
    theta = theta.reshape(sample_sz, n * m)

    ## Initialize Lasso and Fit data
    mini = Lasso(alpha = alpha, fit_intercept = fit_intercept)
    mini.fit(theta, y)
    
    ## Retrieve sparse vector s
    s = mini.coef_
    
    # Reform the image using sparse vector s with inverse descrete cosine
    reform = fft.idctn(s.reshape(n, m), norm='ortho', axes=[0,1])
    if fit_intercept:
        reform += mini.intercept_ # not sure this is right
    
    #return theta, reformed img, sparse vectors
    return theta, reform, s

def color_reconstruct(img_arr, num_cell, cell_size, sparse_freq, alpha = None) :
    if alpha == None :
        alpha = 1 * 50 / num_cell
    i = 0
    dim = img_arr[:,:,i].shape

    W = V1_weights(num_cell, dim, cell_size, sparse_freq) 
    final = np.zeros(img_arr.shape)

    # with same V1 cells generated, reconstruct images for each of 3 rgb arrays and append to final
    while (i < 3):
        img_arr_pt = img_arr[:,:,i]
        img_arr_pt_dim = img_arr_pt.shape
        n_pt, m_pt = img_arr_pt_dim
        y = generate_Y(W, img_arr_pt)
        W_model = W.reshape(num_cell, n_pt, m_pt)
        theta, reconst, s = reconstruct(W_model, y, alpha)
        final[:,:,i] = reconst
        i+=1
        
    final = np.round(final).astype(int)
    final[final < 0] = 0
    final[final > 255] = 255
    final = final.astype(int)
    return final

        
