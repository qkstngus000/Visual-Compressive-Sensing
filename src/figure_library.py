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

#def generate_image():
    


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
    
    return 0;
if __name__ == "__main__":
    main()