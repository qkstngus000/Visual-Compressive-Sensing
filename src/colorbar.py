import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from compress_sensing_library import *
# Package for importing image representation
from PIL import Image, ImageOps

def remove_unnamed_data(data):
    for index in data:
        if (index == 'Unnamed: 0') :
            data.drop('Unnamed: 0', axis = 1, inplace=True)
    return data

def get_min_error_data(observation, data_df):
    if (observation == 'V1'):
        data_mean_df = data_df.groupby(
        ['num_cell', 'sparse_freq', 'cell_size', 'alp'], as_index=False).mean().drop('rep', axis=1) 

        data_min_df = data_mean_df.sort_values('error').drop_duplicates('num_cell')
        data_min_df = data_min_df.rename(columns={'error': 'min_error'})
        data_merged_df = pd.merge(data_df, data_min_df, 
                                       on=['num_cell', 'sparse_freq', 'cell_size', 'alp'], how='left')
        data_plotting_data = data_merged_df.loc[data_merged_df['min_error'].notnull()]

        data_min_mean_err_df = pd.DataFrame()
        for i in data_mean_df['num_cell'].unique():
            data_temp = data_mean_df.loc[data_mean_df['num_cell'] == i]
            #hyperparameter for each number of cell
            ## Grabbing values by each values
            data_min_mean_err_df = data_min_mean_err_df.append(data_temp.loc[data_temp['error'] == data_temp['error'].min()])

        # Merge two data to extract
        data_min_mean_err_df = data_min_mean_err_df.rename(columns={'error' : 'mean_err'})
        data_merged_df = pd.merge(data_df, data_min_mean_err_df, on = ['num_cell', 'sparse_freq', 'cell_size', 'alp'], how = 'left')
#         data_plotting_data = data_merged_df.loc[data_merged_df['mean_err'].notnull()]
    else :
        data_mean_df = data_df.groupby(
            ['num_cell', 'alp'], as_index=False).mean().drop('rep', axis=1) 

        data_min_df = data_mean_df.sort_values('error').drop_duplicates('num_cell')
        data_min_df = data_min_df.rename(columns={'error': 'min_error'})
        data_merged_df = pd.merge(data_df, data_min_df, 
                                       on=['num_cell', 'alp'], how='left')
        data_plotting_data = data_merged_df.loc[data_merged_df['min_error'].notnull()]

        data_min_mean_err_df = pd.DataFrame()
        for i in data_mean_df['num_cell'].unique():
            data_temp = data_mean_df.loc[data_mean_df['num_cell'] == i]
            #hyperparameter for each number of cell
            ## Grabbing values by each values
            data_min_mean_err_df = data_min_mean_err_df.append(data_temp.loc[data_temp['error'] == data_temp['error'].min()])

        # Merge two data to extract
        data_min_mean_err_df = data_min_mean_err_df.rename(columns={'error' : 'mean_err'})
        data_merged_df = pd.merge(data_df, data_min_mean_err_df, on = ['num_cell', 'alp'], how = 'left')
        data_plotting_data = data_merged_df.loc[data_merged_df['mean_err'].notnull()]
      
    return data_min_mean_err_df



def process_data(method, img, V1_file=None, gaussian_file=None, pixel_file=None):
    root = search_root()
    img_nm = img.split('.')[0]
    if (V1_file==None or gaussian_file==None or pixel_file==None):
        print("All three files required to generate figure")
        sys.exit(0)
        
    load_V1 = "{root}/result/{method}/{img_nm}/V1/{file}".format(root = root, method = method, img_nm = img_nm, file = V1_file)
    load_gaussian = "{root}/result/{method}/{img_nm}/gaussian/{file}".format(root = root, method = method, img_nm = img_nm, file = gaussian_file)
    load_pixel = "{root}/result/{method}/{img_nm}/pixel/{file}".format(root = root, method = method, img_nm = img_nm, file = pixel_file)
    
    obs_dict= {'V1': pd.read_csv(load_V1),
               'gaussian': pd.read_csv(load_gaussian), 
               'pixel': pd.read_csv(load_pixel), 
              }
    # Remove unnecessary index column in pandas
    for obs, file in obs_dict.items():
        obs_dict.update({obs: remove_unnamed_data(file)})
        
    for obs, file in obs_dict.items():
        obs_dict.update({obs: get_min_error_data(obs, file)})
        print(obs_dict.get(obs))
    
    
    return obs_dict

def main():
    process_data('dct', 'cameraman.tif', 'black_param_Mon_May_22_21_44_11_2023.csv', 'black_param_Sat_May_20_18_56_44_2023.csv', 'black_param_Sat_May_20_18_40_40_2023.csv')
    
    
main()