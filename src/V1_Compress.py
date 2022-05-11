import numpy as np

from src.structured_random_features.src.models.weights import V1_weights

# Packages for fft and fitting data
from scipy import fftpack as fft
from sklearn.linear_model import Lasso

def generateY(W, img):
    n, m = img.shape
    y = W @ img.reshape(n * m, 1)
    return y

def generateV1Variables(numCell, cellSize, sparseFreq, img):
    # Get size of image
    dim = np.asanyarray(img).shape
    n, m = dim
    # Store generated V1 cells in W
    W = V1_weights(numCell, dim, cellSize, sparseFreq) 
    
    # Retrieve y from W @ imgArr
    y = W @ img.reshape(n*m, 1)

    # Resize W to shape (numCell, height of image, width of image) for fetching into function
    W = W.reshape(numCell, dim[0], dim[1])
    return W, y

def compress(W, y, alpha):
    sampleSz, n, m = W.shape
    
    ## WΨ
    theta = fft.dctn(W.reshape(sampleSz, n, m), norm = 'ortho', axes = [1, 2])
    theta = theta.reshape(sampleSz, n * m)

    ## Initialize Lasso and Fit data
    mini = Lasso(alpha = alpha)
    mini.fit(theta, y)
    
    ## Retrieve sparse vector s
    s = mini.coef_
    
    # Reform the image using sparse vector s with inverse descrete cosine
    reform = fft.idctn(s.reshape(n, m), norm='ortho', axes=[0,1])
    
    #return theta, reformed img, sparse vectors
    return theta, reform, s