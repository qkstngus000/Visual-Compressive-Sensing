import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import time
import os.path
from src.compress_sensing import *
from src.utility import *
from src.args import *
# Package for importing image representation
from PIL import Image, ImageOps


def show_reconstruction_error(img_arr, reconst, method,
                              observation, num_cell, img_name): 
    ''' 
    Display the reconstructed image along with pixel error and a colorbar.
    
    Parameters
    ----------
    img_arr : numpy array 
        Contains the pixel values for the original image

    reconst : numpy array 
        Containing the pixel values for the reconstructed image

    method : String
        Method used for the reconstruction.
        Possible methods are ['dct', 'dwt']

    observation : String
        Observation used to collect data for reconstruction
        Possible observations are ['pixel', 'gaussian', 'V1']

    num_cell : Integer
        Number of blobs that will be used to be determining 
        which pixels to use.

    img_name : String
        Name of the original image file (e.g. "Peppers")
    '''

    # setup figures and axes
    # NOTE: changing figsize here requires you to rescale the colorbar as well
    ## --adjust the shrink parameter to fit.
    fig, axis = plt.subplots(1, 2, figsize = (8, 8))
    plt.tight_layout()

    # prepare the reconstruction axis
    axis[0].set_title(f"{observation} Reconst: {num_cell} cell")
    axis[0].axis('off')

    # prepare the observation error axis
    axis[1].set_title(f"{observation} Error: {num_cell} cells")
    axis[1].axis('off')
    
    # calculate error for RGB images
    if (len(img_arr.shape) == 3):
        axis[0].imshow(reconst, vmin = 0, vmax = 255)
        vmax = ((img_arr - reconst)**2).mean(axis = 2)
        vmax = vmax.max() if vmax.max() < 255 else 255
        err = axis[1].imshow(((img_arr - reconst)**2).mean(axis = 2),
                             'Reds', vmin = 0, vmax = vmax)

    # calculate error for Grayscaled images
    else :
        axis[0].imshow(reconst, cmap='gray', vmin = 0, vmax = 255)
        vmax = img_arr - reconst
        vmax = vmax.max() if vmax.max() < 255 else 255
        err = axis[1].imshow((img_arr - reconst), 'Reds', vmin = 0, vmax = vmax)


    # apply colorbar -- NOTE : if figsize is not (8, 8) then shrink value must be changeed as well
    cbar = fig.colorbar(err, ax=axis, shrink = 0.363, aspect=10)
    cbar.set_label("Error")



def error_vs_num_cell(img, method, pixel_file=None, gaussian_file=None,
                          V1_file=None, data_grab = 'auto') :
    ''' 
    Generate figure that compares which method gives the best minimum error
    
    Parameters
    ----------
    img : String
        The name of image file.
       
    method : String
        Basis the data file was worked on. 
        Currently supporting dct and dwt (discrete cosine/wavelet transform).
    
    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
        Required for plotting.
    
    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
        Required for plotting.
    
    V1_file : String
        V1 observation data file from hyperparameter sweep.
        Required for plotting.
    
    data_grab : String
        With structured path, decides to grab all three data files 
        automatically or manually. Currently not implemented.
        ['auto', 'manual'].
    '''
    img_nm = img.split('.')[0]
    
    if None in [pixel_file, gaussian_file, V1_file] and data_grab == 'manual': 
        print("All observation data file must be given")    
        sys.exit(0)
    
    #Pre-processing data to receive
    data = process_result_data(img, method, 'num_cell', pixel_file, gaussian_file, V1_file)
    plt.xticks(data['V1'][0]['num_cell'])
    plt.xlabel('num_cell')
    print(data)
    title = f"Num_Cell_Vs_Error_{img_nm}_"
    plt.title(title.replace('_', ' '))
    plt.legend(['V1', 'Pixel', 'Gaussian'], loc = 'best')
    
    for obs, plot in data.items():
        sns.lineplot(data = plot[0], x = 'num_cell', y = 'error', label = obs)
        plt.plot(plot[1]['num_cell'], plot[1]['min_error'], 'r.')
    plt.legend(loc = 'best')

def error_vs_alpha(img, method, pixel_file, gaussian_file, V1_file, save = False):
    ''' 
    Generate figure that compares various alpha LASSO penalty and how it affects
    the error of the reconstruction among three different observations. 
    
    Parameters
    ----------
    img : String
        Name of the image that is used by sweeped data
        
    method : String
        Basis the data file was worked on. 
        Currently supporting dct and dwt (discrete cosine/wavelet transform).
    
    pixel_file : String
        Pixel observation data file from hyperparameter sweep.
        Required for plotting.

    gaussian_file : String
        Gaussian observation data file from hyperparameter sweep.
        Required for plotting.

    V1_file : String
        V1 observation data file from hyperparameter sweep.
        Required for plotting.

    save : boolean
        Determines if the image will be saved.
    '''

    
    img_nm = img.split('.')[0]
    if None in [pixel_file, gaussian_file, V1_file]:
        print("Currently all file required")
        sys.exit(0)
    
    if None in [pixel_file, gaussian_file, V1_file] and data_grab == 'manual': 
        print("All observation data file must be given")    
        sys.exit(0)

    #Pre-processing data to receive
    data = process_result_data(img, method, 'alp', pixel_file, gaussian_file, V1_file)
    print(data)
    
    plt.xticks(data['V1'][0]['alp'])
    plt.xlabel('alpha')
    title = f"Alpha_Vs_Error_{img_nm}_"
    plt.title(title.replace('_', ' '))
    plt.legend(['V1', 'Pixel', 'Gaussian'], loc = 'best')
    plt.xscale('log')
    for obs, plot in data.items():
        sns.lineplot(data = plot[0], x = 'alp', y = 'error', label = obs)
        plt.plot(plot[1]['alp'], plot[1]['min_error'], 'r.')
        if obs == 'V1':
            sizes = list(plot[1]['cell_size'])  
            freqs = list(plot[1]['sparse_freq'])
            alphas = list(plot[1]['alp'])
            errors = list(plot[1]['min_error'])
            for i, err in  enumerate(errors):  
                plt.annotate(f'cell_size = {sizes[i]}, sparse_freq = {freqs[i]}',
                      (alphas[i], err))
    plt.legend(loc = 'best')
    
def colorbar_live_reconst(method, img_name, observation, color, dwt_type, level,
                          alpha, num_cells, cell_size, sparse_freq):
    '''
    Generates a reconstruction and error figure for desired parameters.

    Parameters
    ---------
    method : String
        Basis the data file was worked on. 
        Currently supporting dct and dwt (discrete cosine/wavelet transform).

    img_name : String
        The name of image file to reconstruct from.
            
    observation : String
        Observation used to collect data for reconstruction
        Possible observations are ['pixel', 'gaussian', 'V1']
        
    color : bool
        Indicates if the image working on is color image or black/white image
        Possible colors are [True, False]
    
    dwt_type : String
        Type of dwt method to be used.
        See pywt.wavelist() for all possible dwt types.
        
    level : int
        Level of signal frequencies for dwt 
        Better to be an integer in between [1, 4].
        
    alpha : float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents.

    num_cells : int
        Number of blobs that will be used to be determining 
        which pixels to grab and use.
    
    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training.

    sparse_freq : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. Affect the data training
    '''

    fixed_weights = True
    filter_dim = (30, 30)
    img_arr = process_image(img_name, color, False)
    print(f"Image \"{img_name}\" loaded.") 
    reconst = large_img_experiment(
        img_arr, num_cells, cell_size, sparse_freq, filter_dim, alpha, method,
        observation, level, dwt_type, fixed_weights, color) 
    show_reconstruction_error(img_arr, reconst, method, observation,
                   num_cells, img_name.split('.')[0])

def main():
    fig_type, args, save = parse_figure_args()
    if fig_type == 'colorbar' :
      method, img_name, observation, color, dwt_type, level, alpha, num_cells,\
          cell_size, sparse_freq = args
      colorbar_live_reconst(
          method, img_name, observation, color, dwt_type, level,
          alpha, num_cells, cell_size, sparse_freq)
      if save:
          save_reconstruction_error(img_name, method, observation)
    elif fig_type == 'num_cell':
        img_name, method, pixel, gaussian, v1, data_grab = args
        error_vs_num_cell(img_name, method, pixel,
                              gaussian, v1, data_grab)
        if save:
            save_num_cell(img_name, pixel, gaussian, v1, method)
    elif fig_type == 'alpha':
        img_name, method, pixel, gaussian, v1, data_grab = args
        error_vs_alpha(img_name, method, pixel, gaussian, v1, data_grab)
        if save:
            save_alpha(img_name, pixel, gaussian, v1, method)
    if not save:
        plt.show()

if __name__ == "__main__":
    main()
