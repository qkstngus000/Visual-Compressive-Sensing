import numpy as np

from src.structured_random_features.src.models.weights import V1_weights

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
    n, m = img.shape
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

def compress(W, y, alpha):
    sample_sz, n, m = W.shape
    
    ## WΨ
    theta = fft.dctn(W.reshape(sample_sz, n, m), norm = 'ortho', axes = [1, 2])
    theta = theta.reshape(sample_sz, n * m)

    ## Initialize Lasso and Fit data
    mini = Lasso(alpha = alpha)
    mini.fit(theta, y)
    
    ## Retrieve sparse vector s
    s = mini.coef_
    
    # Reform the image using sparse vector s with inverse descrete cosine
    reform = fft.idctn(s.reshape(n, m), norm='ortho', axes=[0,1])
    
    #return theta, reformed img, sparse vectors
    return theta, reform, s
