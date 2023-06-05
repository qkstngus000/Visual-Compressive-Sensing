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

def error_colorbar(img_arr, reconst): 
    if (len(img_arr.shape) == 3):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8), dpi=200)
        fig.set_figheight(7)
        fig.set_figwidth(7)
        plt.tight_layout(h_pad=1)

        ax1.imshow(reconst, vmin = 0, vmax = 255)
        ax1.set_title("Reconst: {num_cell} cell, filter size {f_n}X{f_m}".
                      format(num_cell = num_cell, f_n = filt_dim[0], f_m = filt_dim[1]))
        ax1.axis('off')

        err = ax2.imshow(((img_arr - reconst)**2).mean(axis = 2), 'Reds', vmin = 0, vmax = 255)
        ax2.set_title("Error: {num_cell} cells, filter size {f_n}X{f_m}".
                      format(num_cell = num_cell, f_n = filt_dim[0], f_m = filt_dim[1]))
        ax2.axis('off')

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(err, cax = cax2)
        # ax2.set_aspect('equal')
        # save_path = fig_save_path('peppers', 'dct', 'gaussian/filter_Reconst', "{f_n}X{f_m}_filter_{num_cell}_cell".
        #             format(f_n = filt_dim[0], f_m = filt_dim[1], num_cell = num_cell))
        # fig.savefig(save_path, dpi = 300,  bbox_inches="tight")
        plt.show()
    else :
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 8), dpi=200)
        fig.set_figheight(7)
        fig.set_figwidth(7)
        plt.tight_layout(h_pad=1)

        ax1.imshow(reconst, vmin = 0, vmax = 255)
        ax1.set_title("Reconst: {num_cell} cell, filter size {f_n}X{f_m}".
                      format(num_cell = num_cell, f_n = filt_dim[0], f_m = filt_dim[1]))
        ax1.axis('off')

        err = ax2.imshow((img_arr - reconst), 'Reds', vmin = 0, vmax = 255)
        ax2.set_title("Error: {num_cell} cells, filter size {f_n}X{f_m}".
                      format(num_cell = num_cell, f_n = filt_dim[0], f_m = filt_dim[1]))
        ax2.axis('off')

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(err, cax = cax2)
        ax2.set_aspect('equal')
        
        plt.show()

def get_min_error_V1(img_nm, method, data):
    root = search_root()
    image_name = img_nm.split('.')[0]
    load_V1 = '{root}/result/{method}/{img}/V1/{data}'.format(root = root, method = method, img = image_name, data = data)
    V1_param_df = pd.read_csv(load_V1)
    for index in V1_param_df.columns:
        if (index == 'Unnamed: 0') :
            V1_param_df.drop('Unnamed: 0', axis = 1, inplace=True)
    
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
    
    image = 'image/{img_nm}
    
    
    
    



def main():
    #variables needed
    #print(len(sys.argv))
    #lst = None
    
    return 0;
if __name__ == "__main__":
    main()