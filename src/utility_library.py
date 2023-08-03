import pandas as pd
import seaborn as sns
import time
import os.path
import sys
import os
import re
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
    # Search for the root path
    root = search_root()
    
    method = method.lower()
    observation = observation.upper() if observation.split('/')[0].upper() == 'V1' else observation.lower()
    
    save_nm = save_nm.replace(" ", "_")
    if (observation != 'num_cell_error') :
        if (save_nm[-1] != "_") :
            save_nm = save_nm + "_" 
        save_nm = save_nm + "_".join(str.split(time.ctime().replace(":", "_")))
    
        
    fig_path = os.path.join(root, "figures/{method}/{img_nm}/{observation}".format(
        method = method, img_nm = img_nm, observation = observation))
    Path(fig_path).mkdir(parents=True, exist_ok = True)
    #TODO: add timestamp onto save_nm autometically
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
    # Search for the root path
    root = search_root()
    
    method = method.lower()
    observation = observation.upper() if observation.split('/')[0].upper() == 'V1' else observation.lower()
    
    save_nm = save_nm.replace(" ", "_")
    
    match = re.findall("_hyperparam$", save_nm)
    if (match) : 
        save_nm = save_nm + '.txt'
    else :
        if (save_nm[-1] != "_") :
            save_nm = save_nm + "_" 
        save_nm = save_nm + "_".join(str.split(time.ctime().replace(":", "_"))) + '.csv'  
    print(save_nm)
    sys.exit(0)
    
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
