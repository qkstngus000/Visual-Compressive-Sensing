import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.weights import sensilla_weights, V1_weights, classical_weights

def sensilla_init(layer, lowcut, highcut, decay_coef=np.inf, scale=1, bias=False, seed=None):
    """
    Initialize weights of a Linear layer according to STAs of insect sensilla.
    The bias is turned off by default.
    
    Parameters
    ----------

    layer: torch.nn.Linear layer
        Layer that will be initialized

    lowcut: int
        Low end of the frequency band. 

    highcut: int
        High end of the frequency band.
        
    decay_coef : float, default=np.inf
        controls the how fast the random features decay
        with default value, the weights do not decay
        
    scale: float, default=1
        Normalization factor for Tr norm of cov matrix
        
    bias: Bool, default=False
        The bias of the Linear layer
    
    seed : int, default=None
        Used to set the seed when generating random weights.
    
    """
    classname = layer.__class__.__name__
    assert classname.find('Linear') != -1,'This init only works for Linear layers'
    out_features, in_features = layer.weight.shape
    sensilla_weight = sensilla_weights(out_features, in_features, lowcut, highcut, decay_coef, scale, seed)
    with torch.no_grad():
        layer.weight.copy_(Tensor(sensilla_weight))
    if bias == False:
        layer.bias = None


def V1_init(layer, size, spatial_freq, center=None, scale=1., bias=False, seed=None, tied=False):
    """
    Initialize weights of a Conv2d layer according to receptive fields of V1.
    The bias is turned off by default.
    
    Parameters
    ----------
    layer: torch.nn.Conv2d layer
        Layer that will be initialized
        
    size : float
        Determines the size of the random weights

    spatial_freq : float
        Determines the spatial frequency of the random weights 

    center: tuple of shape (2, 1), default = None
        Location of the center of the random weights
        With default value, the centers uniformly cover the RF space

    scale: float, default=1
        Normalization factor for Tr norm of cov matrix
        
    bias: Bool, default=False
        The bias of the convolutional layer

    seed : int, default=None
        Used to set the seed when generating random weights.
    """
    classname = layer.__class__.__name__
    assert classname.find('Conv2d') != -1, 'This init only works for Conv layers'

    out_channels, in_channels, xdim, ydim = layer.weight.shape
    data = layer.weight.data.numpy().copy()
    # same weights for each channel
    if tied:
        W =  V1_weights(out_channels, (xdim, ydim),
                        size, spatial_freq, center, scale, seed=seed)
    for chan in range(in_channels):
        if not tied:
            W =  V1_weights(out_channels, (xdim, ydim),
                            size, spatial_freq, center, scale, seed=seed)
        data[:, chan, :, :] = W.reshape(out_channels, xdim, ydim)
    data = Tensor(data)
    with torch.no_grad():
        layer.weight.copy_(data)

    if bias == False:
        layer.bias = None


def classical_init(layer, scale=1, bias=False, seed=None):
    """
    Inialize weights of a Linear layer or convolutional layer according to
    GP with diagonal covariance. The bias is turned off by default.
    
    Parameters
    ----------
    
    layer: torch.nn.Linear layer
        Layer that will be initialized
        
    scale: float, default=1
        Normalization factor for Tr norm of cov matrix
        
    bias: Bool, default=False
        The bias of the Linear layer
    
    seed : int, default=None
        Used to set the seed when generating random weights.

    """
    classname = layer.__class__.__name__
    assert classname.find('Linear') != -1 or classname.find('Conv2d') != -1, 'This init only works for Linear or Conv layers' 

    if classname.find('Linear') == 1: 
        in_features, out_features = layer.weight.shape
        classical_weight = classical_weights(out_features, in_features, scale, seed)
        data = Tensor(classical_weight)
        with torch.no_grad():
            layer.weight.copy_(data)
        
    elif classname.find('Conv2d') == 1:
        out_channels, in_channels, xdim, ydim = layer.weight.shape
        data = layer.weight.data.numpy().copy()
        for chan in range(in_channels):
            W = classical_weights(out_channels, (xdim, ydim), scale, seed=seed)
            data[:, chan, :, :] = W.reshape(out_channels, xdim, ydim)
        data = Tensor(data)
        with torch.no_grad():
            layer.weight.copy_(data)
        
    if bias == False:
        layer.bias = None
        # else, do nothing

