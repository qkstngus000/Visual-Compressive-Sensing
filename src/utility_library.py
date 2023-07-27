import pandas as pd
import seaborn as sns
import time
import os.path
import sys
import os
import matplotlib.pyplot as plt
sys.path.append("../")
from src.compress_sensing_library import *

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
        
    if (observation.split('/')[0] != 'v1' or observation.split('/')[0] != 'V1') :
        observation = observation.lower()
    else :
        observation = observation.upper()
        
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
        print("upper triggered: " + observation.split('/')[0])
        observation = observation.upper()
        
    else :
        print("lower triggered: " + observation)
        observation = observation.lower()
        
        
    result_path = os.path.join(root, "result/{method}/{img_nm}/{observation}".format(
        method = method, img_nm = img_nm, observation = observation))
    Path(result_path).mkdir(parents=True, exist_ok = True)
    
    return os.path.join(result_path, "{save_nm}.csv".format(save_nm = save_nm))


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
    color = ['-c', 'color']
    if mode.lower() not in color:
        img_path = ImageOps.grayscale(img_path)
    if visibility:
        if mode.lower() not in color:
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
