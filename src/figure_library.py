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

# Package for importing image representation
from PIL import Image, ImageOps

def remove_unnamed_data(data):
    for index in data:
        if (index == 'Unnamed: 0') :
            data.drop('Unnamed: 0', axis = 1, inplace=True)
    return data


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
        err = axis[1].imshow(((img_arr - reconst)**2).mean(axis = 2), 'Reds', vmin = 0, vmax = 255)

    # calculate error for Grayscaled images
    else :
        axis[0].imshow(reconst, cmap='gray', vmin = 0, vmax = 255)
        err = axis[1].imshow((img_arr - reconst), 'Reds', vmin = 0, vmax = 255)

    # apply colorbar -- NOTE : if figsize is not (8, 8) then shrink value must be changeed as well
    cbar = fig.colorbar(err, ax=axis, shrink = 0.363, aspect=10)
    cbar.set_label("Error")

    # save image to outfile if desired, else display to the user
    if save_img == True:
        outfile = fig_save_path(img_name, "dct", observation, "colorbar")
        plt.savefig(outfile, dpi = 300, bbox_inces = "tight")
    else:
        plt.show()

def get_min_error_V1(img_nm, method, observation, data):
    
    V1_param_mean_df = V1_param_df.groupby(
    ['num_cell', 'sparse_freq', 'cell_size', 'alp'], as_index=False).mean().drop('rep', axis=1) 

    V1_param_min_df = V1_param_mean_df.sort_values('error').drop_duplicates('num_cell')
    V1_param_min_df = V1_param_min_df.rename(columns={'error': 'min_error'})
    V1_merged_df = pd.merge(V1_param_df, V1_param_min_df, 
                                   on=['num_cell', 'sparse_freq', 'cell_size', 'alp'], how='left')
    V1_plotting_data = V1_merged_df.loc[V1_merged_df['min_error'].notnull()]

    V1_min_mean_err_df = pd.DataFrame()
    for i in V1_param_mean_df['num_cell'].unique():
        V1_temp = V1_param_mean_df.loc[V1_param_mean_df['num_cell'] == i]
        #hyperparameter for each number of cell
        ## Grabbing values by each values
        V1_min_mean_err_df = V1_min_mean_err_df.append(V1_temp.loc[V1_temp['error'] == V1_temp['error'].min()])
        
    # Merge two data to extract
    V1_min_mean_err_df = V1_min_mean_err_df.rename(columns={'error' : 'mean_err'})
    V1_merged_df = pd.merge(V1_param_df, V1_min_mean_err_df, on = ['num_cell', 'sparse_freq', 'cell_size', 'alp'], how = 'left')
    V1_plotting_data = V1_merged_df.loc[V1_merged_df['mean_err'].notnull()]
    print(V1_param_min_df)


def main():
    #variables needed
    #print(len(sys.argv))
    #lst = None
    img_arr = process_image("tree_part1.jpg", "color", False)
    reconst = np.load("tree_part1_reconst.npy")
    method = "dct"
    observation = "V1"
    num_cell = "500"
    error_colorbar(img_arr, reconst, method, observation, num_cell, "peppers", False)

    return 0;
if __name__ == "__main__":
    main()