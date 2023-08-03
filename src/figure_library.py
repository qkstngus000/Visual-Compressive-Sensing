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
    ''' Remove unnecessary data column that stores index 
    
    Parameters
    ----------
    data : pandas dataframe
        Dataframe Storing reconstruction hyperparameters and errors
            
    Returns
    ----------
    data : pandas dataframe
        Dataframe Storing reconstruction hyperparameters and errors, but without unnecessary column
    
    '''
    for index in data:
        if (index == 'Unnamed: 0') :
            data.drop('Unnamed: 0', axis = 1, inplace=True)
    return data

def process_result_data(img_file, method, pixel_file=None, gaussian_file=None, V1_file=None):
    ''' Open 3 csv data files, make it as pandas dataframe, remove unnecessary column, find the plotting data with minimum mean error for each of num_cell
    
    Parameters
    ----------
    img_file : String
        The name of image file that will be worked on
        
    method : String
        Basis the data file was worked on. Currently supporting dct (descrete cosine transform) and dwt (descrete wavelet transform)
        
    pixel_file : String
        pixel observation data file from hyperparameter sweep that is needed to plot
    
    gaussian_file : String
        gaussian observation data file from hyperparameter sweep that is needed to plot
    
    V1_file : String
        V1 observation data file from hyperparameter sweep that is needed to plot
        
    Returns
    ----------
    obs_dict : python dictionary
        Dictionary that contains ['V1', 'gaussian', 'pixel'] as a key and [0]th value storing plotting data with [1]st data containing minimum mean error parameter for each num_cell
    
    '''
    root = search_root()
    img_nm = img_file.split('.')[0]
    
    # TODO: Currently all three files required, but thinking if plot can be generated with just one or two files as well
    if (V1_file==None or gaussian_file==None or pixel_file==None):
        print("All three files required to generate figure")
        sys.exit(0)
        
        
    load_V1 = "{root}/result/{method}/{img_nm}/V1/{file}".format(
        root = root, method = method, img_nm = img_nm, file = V1_file)
    load_gaussian = "{root}/result/{method}/{img_nm}/gaussian/{file}".format(
        root = root, method = method, img_nm = img_nm, file = gaussian_file)
    load_pixel = "{root}/result/{method}/{img_nm}/pixel/{file}".format(
        root = root, method = method, img_nm = img_nm, file = pixel_file)
    
    obs_dict= {'V1': pd.read_csv(load_V1),
               'gaussian': pd.read_csv(load_gaussian), 
               'pixel': pd.read_csv(load_pixel), 
              }
    # Remove unnecessary index column in pandas
    for obs, file in obs_dict.items():
        obs_dict.update({obs: remove_unnamed_data(file)})
        
    for obs, file in obs_dict.items():
        obs_dict.update({obs: get_min_error_data(method, obs, file)})
    
    return obs_dict
    


def error_colorbar(img_arr, reconst, observation, num_cell): 
    if (len(img_arr.shape) == 3):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))
        plt.tight_layout()
        ax1.imshow(reconst, cmap='gray', vmin = 0, vmax = 255)
        ax1.set_title("{observation} Reconst: {num_cell} cell".format(observation=observation, num_cell = num_cell))
        ax1.axis('off')

        err = ax2.imshow(((img_arr - reconst)**2).mean(axis = 2), 'Reds', vmin = 0, vmax = 255)
        ax2.set_title("{observation}Error: {num_cell} cells".format(observation = observation, num_cell = num_cell))
        ax2.axis('off')

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(err, cax = cax2)
        # ax2.set_aspect('equal')
        # save_path = fig_save_path('peppers', 'dct', 'gaussian/filter_Reconst', "{f_n}X{f_m}_filter_{num_cell}_cell".
        #             format(f_n = filt_dim[0], f_m = filt_dim[1], num_cell = num_cell))
        # fig.savefig(save_path, dpi = 300,  bbox_inches="tight")
#         plt.show()
    else :
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8))
        plt.tight_layout()

        ax1.imshow(reconst, vmin = 0, vmax = 255)
        ax1.set_title("{observation} Reconst: {num_cell} cell".format(observation=observation, num_cell = num_cell))
        ax1.axis('off')

        err = ax2.imshow((img_arr - reconst), 'Reds', vmin = 0, vmax = 255)
        ax2.set_title("{observation} Reconst: {num_cell} cell".format(observation=observation, num_cell = num_cell))
        ax2.axis('off')

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(err, cax = cax2)
        ax2.set_aspect('equal')
    return fig, ax1, ax2

def get_min_error_data(method, observation, data_df):
    ''' Retrieve plotting data and minimum error parameter to be returned
    
    Parameters
    ----------
    method : String
        Basis the data file was worked on. Currently supporting dct (descrete cosine transform) and dwt (descrete wavelet transform)
        
    observation : String
        Observation technique that are going to be used to collet sample for reconstruction.
    
    data_df : pandas dataframe
        Dataframe Storing reconstruction hyperparameters and errors
            
    Returns
    ----------
    data_plotting_data : pandas dataframe
        Dataframe storing all hyperparameters that have the minimum error
    
    data_min_df : pandas dataframe
        Dataframe storing hyperparameters that have minimum mean errors for each num_ell, but without unnecessary column
    
    '''
    param_list = []
    
    # V1 observation takes two more parameter
    if (observation.upper() == 'V1'):
        # Check the method first to check what parameter it has to deal with, since dwt takes two more parameters
        if (method.lower() == 'dct'):
            param_list = ['num_cell', 'sparse_freq', 'cell_size', 'alp']
        elif (method.lower() == 'dwt') :
            param_list = ['num_cell', 'sparse_freq', 'cell_size', 'alp', 'lv']
            
    else :
        if (method == 'dct'):
            param_list = ['num_cell', 'alp']
        elif (method == 'dwt') :
            param_list = ['num_cell', 'alp', 'lv']
    
    # For each hyperparameter, gives mean of its own repetition
    data_mean_df = data_df.groupby(param_list, as_index=False).mean().drop('rep', axis=1) 
    
    # Grab the lowest mean error from each number of cell
    data_min_df = data_mean_df.sort_values('error').drop_duplicates('num_cell')
    data_min_df = data_min_df.rename(columns={'error': 'min_error'})
    
    # Mark the hyperparameter that gives lowest mean error to whole dataset
    data_merged_df = pd.merge(data_df, data_min_df, on=param_list, how='left')
    
    # Grab hyperparameters that was marked
    data_plotting_data = data_merged_df.loc[data_merged_df['min_error'].notnull()]

    return (data_plotting_data, data_min_df)

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
        sns.lineplot(data = plot[0], x = 'num_cell', y = 'error', palette='Accent', label = obs)
        plt.plot(plot[1]['num_cell'], plot[1]['min_error'], 'r.')
    plt.legend(loc = 'best')
    if save :
        # for its save name, the name of file order is pixel -> gaussian -> V1 
        save_nm = pixel_file.split('.')[0] + '_' + gaussian_file.split('.')[0] + '_' + V1_file.split('.')[0]
        save_path = fig_save_path(img_nm, method, 'num_cell_error', save_nm)
        plt.savefig(save_path, dpi = 200)
        
    plt.show()
    
def main():
    #variables needed
    #print(len(sys.argv))
    #lst = None
    
    return 0;
if __name__ == "__main__":
    main()