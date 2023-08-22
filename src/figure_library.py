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


def error_colorbar(img_arr, reconst, method, observation, num_cell, img_name,
                   save_img = False): 
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



def num_cell_error_figure(img, method, pixel_file=None, gaussian_file=None,
                          V1_file=None, data_grab = 'auto', save = False) :
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
    
    save : bool
        Save data into specified path.
        [True, False]
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

def alpha_error(img, method, pixel_data, gaussian_data, V1_data, save = False):
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
    
    pixel_data : String
        Pixel observation data file from hyperparameter sweep.
        Required for plotting.

    gaussian_data : String
        Gaussian observation data file from hyperparameter sweep.
        Required for plotting.

    V1_data : String
        V1 observation data file from hyperparameter sweep.
        Required for plotting.

    save : boolean
        Determines if the image will be saved.
    '''
    if None in [pixel_data, gaussian_data, V1_data]:
        print("Currently all file required")
        sys.exit(0)
    
    title = ''
    
    # Preprocess data not to have 
    pixel_df = remove_unnamed_data(pd.read_csv(pixel_data))
    gaussian_df = remove_unnamed_data(pd.read_csv(gaussian_data))
    V1_df = remove_unnamed_data(pd.read_csv(V1_data))
    
    num_cell_list = V1_df['num_cell'].unique()
    
    for num_cell in num_cell_list :
        # In order to bring fixed cell_size and sparse_frequency, bring parameter that has median error value
        V1_df_mean = V1_df.loc[V1_df["num_cell"] == num_cell].groupby(list(V1_df.columns[1:-1]), 
                                               as_index = False).mean().drop('rep', axis=1)
        median_col = V1_df_mean.loc[V1_df_mean['error'] == V1_df_mean['error'].median()]

        # Depending on the basis used (dct / dwt) add lv parameter
        if (method.lower() == 'dct') :
            cell_size, sparse_freq = V1_df_mean.loc[V1_df_mean['error'] == V1_df_mean['error'].
                                                    median()][['cell_size', 'sparse_freq']].values.squeeze()
            V1_df_mod = V1_df.loc[(V1_df['cell_size'] == cell_size) & 
                                  (V1_df['sparse_freq'] == sparse_freq)]
            title=r"$\alpha$_Error for {cell} cells (cell_size: {cell_size}, sparse_freq: {sparse_freq})".format(
                cell = num_cell, cell_size = cell_size, sparse_freq = sparse_freq)
        else :
            cell_size, sparse_freq, lv = V1_df_mean.loc[V1_df_mean['error'] == V1_df_mean['error'].
                                                    median()][['cell_size', 'sparse_freq', 'lv']].values.squeeze()
            V1_df_mod = V1_df.loc[(V1_df['cell_size'] == cell_size) & 
                                  (V1_df['sparse_freq'] == sparse_freq) & 
                                  (V1_df['lv'] == lv)]
            title=r"$\alpha$_Error for {cell} cells \
            (cell_size: {cell_size}, sparse_freq: {sparse_freq}, lv: {lv})".format(
                cell = num_cell, cell_size = cell_size, sparse_freq = sparse_freq, lv = lv)

        fig = sns.relplot(data = V1_df_mod, x = 'alp', y = 'error', kind='line', palette='Accent', 
                          legend = True, label = 'V1')


        fig.map(sns.lineplot, x = 'alp', y = 'error', data = pixel_df.loc[pixel_df["num_cell"] == num_cell], 
                label= 'pixel', color = 'red', 
                legend = True)
        fig.map(sns.lineplot, x = 'alp', y = 'error', data = gaussian_df.loc[gaussian_df["num_cell"] == num_cell], 
                label= 'gaussian', color = 'green', 
                legend = True)
        fig.set(title = title)
        fig.add_legend(title='Observation', loc = 'right')
        fig.set(xscale='log')
        fig.set(yscale='log')
        plt.xlabel(r"$\alpha$")
        
        # Save the figure
        if save :
            path = fig_save_path(img, method, 'combined', title)
            plt.savefig(path, dpi = 200)
        plt.show()

def colorbar_live_reconst(method, img_name, observation, mode, dwt_type, level,
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
        
    mode : String
        Mode to reconstruct image ['color' or 'black']
    
    dwt_type : String
        Type of dwt method to be used.
        See pywt.wavelist() for all possible dwt types.
        
    level : int
        Level of signal frequencies for dwt -- should be an integer in [1, 4].
        
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
    rand_weight = False
    filter_dim = (30, 30)
    img_arr = process_image(img_name, mode, False)
    print(f"Image \"{img_name}\" loaded.") 
    reconst = large_img_experiment(img_arr, num_cells, cell_size, sparse_freq, filter_dim, alpha, method, observation, level, dwt_type, rand_weight, mode) 
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
