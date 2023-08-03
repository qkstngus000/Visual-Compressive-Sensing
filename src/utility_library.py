import pandas as pd
import seaborn as sns
import time
import os.path
import sys
import os
import matplotlib.pyplot as plt
sys.path.append("../")
from src.compress_sensing_library import *

# Package for importing image representation
from PIL import Image, ImageOps

def search_root():
    ''' Search for the root directory of the project
    
    Parameters
    ----------
        
    Returns
    ----------
    root : path
        Return absolute paths for the project
    
    '''
    back_path = './'
    root = Path(os.path.abspath('./'))
    while not root.match('*/research'):
        back_path += '../'
        root = Path(os.path.abspath(back_path))
    return root

def fig_save_path(img_nm, method, observation, save_nm):
    ''' Gives absolute paths for generated figures to be saved to have organized folder structure. Figure will be saved under figure directory and its format is set up to be png file.If folder path does not exist with the parameter given, then create a new path
    
    Parameters
    ----------
        img_nm : String
            Name of the image file data used 
        
        method : String
            Method used for the reconstruction.
            Possible methods are ['dct', 'dwt']
        
        observation : String
            Observation used to collect data for reconstruction
            Possible observations are ['pixel', 'gaussian', 'V1']
        
        save_nm : String
            Name of the file that it will be saved to
        
    Returns
    ----------
    path : path
        Return absolute path including its file name
    
    '''
    
    save_nm = save_nm.replace(" ", "_")
    method = method.lower()
    
    # Search for the root path
    root = search_root()
        
    if (observation.split('/')[0] == 'v1' or observation.split('/')[0] == 'V1') :
        observation = observation.upper()
    else :
        observation = observation.lower()
        
    # Except for the combined num_cell vs error figure, add time that figure was created for its uniqueness
    if (observation.lower() != 'num_cell_error') :
        if save_nm[-1] != '_' :
            save_nm = save_nm + "_"
        save_nm = save_nm + "_".join(str.split(time.ctime().replace(":", "_")))
        
    fig_path = os.path.join(root, "figures/{method}/{img_nm}/{observation}".format(
        method = method, img_nm = img_nm, observation = observation))
    Path(fig_path).mkdir(parents=True, exist_ok = True)
    return os.path.join(fig_path, "{save_nm}.png".format(save_nm = save_nm))

def data_save_path(img_nm, method, observation, save_nm): 
    ''' Gives absolute paths for collected data to be saved to have organized folder structure. File will be saved under result directory and its format is set up to be csv file. If folder path does not exist with the parameter given, then create a new path
    
    Parameters
    ----------
        img_nm : String
            Name of the image file data used 
        
        method : String
            Method used for the reconstruction.
            Possible methods are ['dct', 'dwt']
        
        observation : String
            Observation used to collect data for reconstruction
            Possible observations are ['pixel', 'gaussian', 'V1']
        
        save_nm : String
            Name of the file that it will be saved to
        
    Returns
    ----------
    path : path
        Return absolute path including its file name
    
    '''
    save_nm = save_nm.replace(" ", "_")
    method = method.lower()
    
    # Search for the root path
    root = search_root()
    
    if (observation.split('/')[0] == 'v1' or observation.split('/')[0] == 'V1') :
        observation = observation.upper()
    else :
        observation = observation.lower()
        
    if save_nm[-1] != '_' :
        save_nm = save_nm + "_"
    save_nm = save_nm + "_".join(str.split(time.ctime().replace(":", "_")))
        
    result_path = os.path.join(root, "result/{method}/{img_nm}/{observation}".format(
        method = method, img_nm = img_nm, observation = observation))
    Path(result_path).mkdir(parents=True, exist_ok = True)
    
    return os.path.join(result_path, "{save_nm}.csv".format(save_nm = save_nm))

def color():
    ''' Pools for user argument on color mode
    
    Parameters
    ----------
    
    Returns
    ----------
    list : list
        return list of possible user answer for color mode
    '''
    return ['-c', 'c', 'color', '-color']

def process_image(img, mode = 'black', visibility = False):
    ''' Opens image file with given file name and determines whether image file will be color or grayscale image. Show the image if the visibility is set to be True
    
    Parameters
    ----------
        img : String
            Name of the image file data to be opened
        
        mode : String
            Determines whether image data will be grascaled or not
        
        visibility : bool
            Determines whether image will be shown or not to check if image is correctly opened     
        
    Returns
    ----------
    img_arr : array_like
        Return numpy array of image
    
    '''
    root = search_root()
    img_path = Image.open(os.path.join(root, 'image/{img}'.format(img=img)))
    if mode.lower() not in color():
        img_path = ImageOps.grayscale(img_path)
    if visibility:
        if mode.lower() not in color():
            print("Processing grayscale image")
            plt.imshow(img_path, 'gray')
            plt.title("grayscaled_{img}".format(img = img.split(".")[0]))
        else:
            print("Processing color image")
            plt.imshow(img_path)
            plt.title(img.split(".")[0])
        plt.axis('off')
        plt.show()
    img_arr = np.asarray(img_path)
    
    return img_arr


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
