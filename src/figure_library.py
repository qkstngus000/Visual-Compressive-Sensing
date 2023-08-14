import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append("../")

from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import time
import os.path
from src.compress_sensing_library import *
from src.utility_library import *
from src.arg_library import *
# Package for importing image representation
from PIL import Image, ImageOps



def error_colorbar(img_arr, reconst, method, observation, num_cell, img_name, save_img = False): 
    ''' Display the reconstructed image along with pixel error and a colorbar.
    
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
            Number of blobs that will be used to be determining which pixles to grab and use
    
        img_name : String
            Name of the original image file (e.g. "Peppers")
        
        save_img : boolean
            Determines if the image will be saved.
    '''

    # setup figures and axes
    # NOTE: changing figsize here requires you to rescale the colorbar as well --adjust the shrink parameter to fit.
    fig, axis = plt.subplots(1, 2, figsize = (8, 8))
    plt.tight_layout()

    # prepare the reconstruction axis
    axis[0].set_title("{observation} Reconst: {num_cell} cell".format(observation=observation, num_cell = num_cell))
    axis[0].axis('off')

    # prepare the observation error axis
    axis[1].set_title("{observation} Error: {num_cell} cells".format(observation = observation, num_cell = num_cell))
    axis[1].axis('off')
    
    # calculate error for RGB images
    if (len(img_arr.shape) == 3):
        axis[0].imshow(reconst, vmin = 0, vmax = 255)
        vmax = ((img_arr - reconst)**2).mean(axis = 2)
        vmax = vmax.max() if vmax.max() < 255 else 255
        err = axis[1].imshow(((img_arr - reconst)**2).mean(axis = 2), 'Reds', vmin = 0, vmax = vmax)

    # calculate error for Grayscaled images
    else :
        axis[0].imshow(reconst, cmap='gray', vmin = 0, vmax = 255)
        vmax = img_arr - reconst
        vmax = vmax.max() if vmax.max() < 255 else 255
        err = axis[1].imshow((img_arr - reconst), 'Reds', vmin = 0, vmax = vmax)


    # apply colorbar -- NOTE : if figsize is not (8, 8) then shrink value must be changeed as well
    cbar = fig.colorbar(err, ax=axis, shrink = 0.363, aspect=10)
    cbar.set_label("Error")
    # save image to outfile if desired, else display to the user
    if save_img == True:
        outfile = fig_save_path(img_name, "dct", observation, "colorbar")
        plt.savefig(outfile, dpi = 300, bbox_inches = "tight")
    else:
        plt.show()



def num_cell_error_figure(img, method, pixel_file=None, gaussian_file=None, V1_file=None, data_grab = 'auto', save = False) :
    ''' Generate figure that compares which method gives the best minimum error
    
    Parameters
    ----------
    img : String
        the name of image file
       
    method : String
        Basis the data file was worked on. Currently supporting dct (descrete cosine transform) and dwt (descrete wavelet transform)
    
    pixel_file : String
        pixel observation data file from hyperparameter sweep that is needed to plot
    
    gaussian_file : String
        gaussian observation data file from hyperparameter sweep that is needed to plot
    
    V1_file : String
        V1 observation data file from hyperparameter sweep that is needed to plot
    
    data_grab : String
        With structured path, decides to grab all three data file automatically or manually. Currently not implemented
        ['auto', 'manual']
    
    save : bool
        Save data into specified path
        [True, False]
            
    Returns
    ----------
    '''
    img_nm = img.split('.')[0]
    
    if None in [pixel_file, gaussian_file, V1_file] and data_grab == 'manual': 
        print("All observation data file must be given")    
        sys.exit(0)
    
    #Pre-processing data to receive
    data = process_result_data(img, method, pixel_file, gaussian_file, V1_file)
    plt.xticks(data['V1'][0]['num_cell'])
    plt.xlabel('num_cell')
    title = "Num_Cell_Vs_Error_{img}_".format(img = img_nm)
    plt.title(title.replace('_', ' '))
    plt.legend(['V1', 'Pixel', 'Gaussian'], loc = 'best')
    
    for obs, plot in data.items():
        sns.lineplot(data = plot[0], x = 'num_cell', y = 'error', label = obs)
        plt.plot(plot[1]['num_cell'], plot[1]['min_error'], 'r.')
    plt.legend(loc = 'best')
    if save :
        # for its save name, the name of file order is pixel -> gaussian -> V1 
        save_nm = pixel_file.split('.')[0] + '_' + gaussian_file.split('.')[0] + '_' + V1_file.split('.')[0]
        save_path = fig_save_path(img_nm, method, 'num_cell_error', save_nm)
        plt.savefig(save_path, dpi = 200)
        
    plt.show()


def colorbar_live_reconst(method, img_name, observation, mode, dwt_type, level, alpha, num_cells, cell_size, sparse_freq):
    rand_weight = False
    filter_dim = (30, 30)
    img_arr = process_image(img_name, mode, False)
    print(f"Image \"{img_name}\" loaded.") 
    reconst = filter_reconstruct(img_arr, num_cells, cell_size, sparse_freq, filter_dim, alpha, method, observation, level, dwt_type, rand_weight, mode) 
    print(f"Image {img_name} reconstructed. Displaying reconstruction and error.") 
    error_colorbar(img_arr, reconst, method, observation, num_cells, img_name.split('.')[0], False)

    
def main():
    fig_type, args = parse_figure_args()
    if fig_type == 'colorbar' :
      method, img_name, observation, mode, dwt_type, level, alpha, num_cells, cell_size, sparse_freq = args
      colorbar_live_reconst(method, img_name, observation, mode, dwt_type, level, alpha, num_cells, cell_size, sparse_freq)      
    elif fig_type == 'num_cell':
        img_name, method, pixel, gaussian, v1, data_grab, save = args
        num_cell_error_figure(img_name, method, pixel, gaussian, v1, data_grab, save)


if __name__ == "__main__":
    main()