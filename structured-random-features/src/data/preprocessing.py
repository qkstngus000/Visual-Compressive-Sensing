import glob

import numpy as np
import pickle
from scipy.io import loadmat
from scipy import ndimage
from skimage.transform import downscale_local_mean
from torchvision import datasets

def extract_sensilla_STA(input_filepath, output_filepath):
    """
     Construct spike triggered averages from sensilla mechanosensors. We are
     using data provided by Brandon Pratt.
     Neural evidence supports a dual sensory-motor role for insect wings 
     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5597827/

    input_filepath: str
        Location of raw sensilla data

    output_filepath: str
        Output location of processed STAs 
    """
    data_dir = input_filepath + '/sensilla_Pratt/'

    # load stimulus and spike information from sensilla at the base of the
    # wing and in the middle of the wing
    stim = loadmat(data_dir + 'Motor Tip Displacement.mat')['Tip_Signal'].flatten()
    nonbase_spike_filenames = []
    for file_name in glob.glob(data_dir + 'NonBase Spike Trains/NonBase*.mat'):
        nonbase_spike_filenames.append(file_name)
    base_spike_filenames = []
    for file_name in glob.glob(data_dir + 'Base Spike Trains/Spikes*.mat'):
        base_spike_filenames.append(file_name)
    spike_filenames = nonbase_spike_filenames + base_spike_filenames

    # experimental parameters
    sample_rate = 4e4
    sta_time_window = 0.04 # in seconds
    sample_window = int(sample_rate * sta_time_window)
    nTrials = 30

    # Create STA from each cell
    STA = np.empty((0, sample_window))
    for file in spike_filenames:
        spike_mat = loadmat(file)['WN_Repeat_Matrix']
        
        if len(spike_mat.shape) == 2:
            spike_mat = np.expand_dims(spike_mat, axis=2)

        num_trials, num_sensilla = spike_mat.shape[1], spike_mat.shape[2]
        for sensilla in range(num_sensilla):
            spikes_for_sensilla = spike_mat[:, :, sensilla]
            spike_times = np.argwhere(spikes_for_sensilla == 1)[:, 0]
            spike_times = spike_times[spike_times > sample_window]

            STA_for_sensilla = np.zeros(sample_window)
            for time in spike_times:
                STA_for_sensilla += stim[time - sample_window: time]
            STA_for_sensilla /= len(spike_times)

            STA = np.row_stack((STA, STA_for_sensilla))

    output_dict = {'STA': STA, 'sample_rate': 4e4, 'sta_time_window': 0.04}
    with open(output_filepath + '/STA_sensilla.pickle', 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def center_receptive_field(cell_rf, xdim, ydim):
    """
    Centers the receptive field of visual neurons.

    Parameters
    ----------
    cell_rf: array_like (n, d)
        receptive field of a neuron

    xdim: int
        height of the receptive field

    ydim: int
        width of the receptive field

    returns
    -------
    centered_rf: array_like (n, d)
        rf centered to (xdim/2, ydim/2)
    """
    # find center of the receptive field
    center = (int(xdim / 2), int(ydim / 2))
    center_of_mass = ndimage.measurements.center_of_mass(np.abs(cell_rf) ** 4)
    center_of_mass = np.round(center_of_mass).astype('int')

    # translate the center to the middle of the image but wrap around
    centered_rf = np.roll(cell_rf, center[0] - center_of_mass[0], axis=0)
    centered_rf = np.roll(centered_rf, center[1] - center_of_mass[1], axis=1)

    return centered_rf


def extract_V1_rf_whitenoise(input_filepath, output_filepath, centered=True):
    """
    Constructs the V1 receptive fields dataset given by Marius Pachitariu. 44k 
    neurons were shown white noise repeatedly and their RFs were calculated. In the data, 
    the RF sizes are 14 x 36. We center the RFs for analysis.
    
    Parameters
    ----------
    input_filepath: str
        Location of V1 rfs from whitenoise

    output_filepath: str
        output location

    centered: Bool
        Move the receptive fields to the center of the image 

    Returns
    -------
    output_dict: dict
        Dictionary with centered rf & snr for every cell, xdim and ydim of
        the rf 
    """
 
    data_dir = input_filepath + '/rf_whitenoise_Marius.npz'

    # load data
    with np.load(data_dir) as data:
        RF_Marius = data['rf']
        snr = data['snr']
        num_cells, xdim, ydim = RF_Marius.shape
    processed_RF = np.zeros((num_cells, xdim * ydim))
    
    for cell in range(num_cells):
        cell_rf = RF_Marius[cell, :, :]

        # center
        if centered == True:
            centered_rf = center_receptive_field(cell_rf, xdim, ydim)
            processed_RF[cell] = centered_rf.flatten()

        elif centered == False:
            processed_RF[cell] = cell_rf.flatten() 
    
    # save into an output dict and pickle dump
    output_dict = {'rf': processed_RF, 'snr': snr, 'xdim': xdim, 'ydim': ydim}
    with open(output_filepath + '/rf_whitenoise{}.pickle'.format(centered * '_centered'), 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def extract_V1_rf_natural_images(input_filepath, output_filepath, centered=True):
    """
    Loads the V1 receptive fields dataset given by Marius Pachitariu. 70k neurons were
    shown 5000 naturalistic images in random order in 3 trials. In the data, 
    the RF sizes are 24 x 27. We can center them for analysis.
    
    Parameters
    ----------
    input_filepath: str
        Location of V1 rfs from natural images

    output_filepath: str
        output location

    centered: Bool
        Move the receptive fields to the center of the image 

    Returns
    -------
    output_dict: dict
        Dictionary with centered rf & snr for every cell, xdim and ydim of
        the rf 
    """

    data_dir = input_filepath + '/rf_natural_images_Marius.npz'

    # load data
    with np.load(data_dir) as data:
        RF_Marius = data['rf']
        xpos = data['xpos']
        y_pos = data['ypos']
        snr = data['snr']
        num_trials, num_cells, xdim, ydim = RF_Marius.shape
    processed_RF = np.zeros((num_trials, num_cells, xdim * ydim))

    for trial in range(num_trials):
        for cell in range(num_cells):
            cell_rf = RF_Marius[trial, cell]
                
            # center
            if centered == True:
                centered_rf = center_receptive_field(cell_rf, xdim, ydim)
                processed_RF[trial, cell] = centered_rf.flatten()

            elif centered == False:
                processed_RF[trial, cell] = cell_rf.flatten()
    
    # save into an output dict and pickle dump
    output_dict = {'rf': processed_RF, 'snr': snr, 'xdim': xdim, 'ydim': ydim}
    with open(output_filepath + '/rf_natural_images{}.pickle'.format(centered * '_centered'), 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def extract_V1_rf_DHT(input_filepath, output_filepath, centered=True):
    """
    Loads the V1 receptive fields dataset given by Marius Pachitariu. 4k neurons were
    shown hartley basis functions repeatedly and their RFs were calculated. In the data, 
    the RF sizes are 30 x 80. We can center them for analysis.
    
    Parameters
    ----------
    input_filepath: str
        Location of V1 rfs from DHT stimuli

    output_filepath: str
        output location

    centered: Bool
        Move the receptive fields to the center of the image 

    Returns
    -------
    output_dict: dict
        Dictionary with centered rf & snr for every cell, xdim and ydim of
        the rf 
    """

    data_dir = input_filepath + '/rf_DHT_Marius.npz'

    # load data
    with np.load(data_dir) as data:
        RF_Marius = data['rf']
        snr = data['snr']
        xdim, ydim, num_cells = RF_Marius.shape
    processed_RF = np.zeros((num_cells, xdim * ydim))
    
    for cell in range(num_cells):
        cell_rf = RF_Marius[:, :, cell]

       # center
        if centered == True:
            centered_rf = center_receptive_field(cell_rf, xdim, ydim)
            processed_RF[cell] = centered_rf.flatten()

        elif centered == False:
            processed_RF[cell] = cell_rf.flatten()

    # save into an output dict and pickle dump
    output_dict = {'rf': processed_RF, 'snr': snr, 'xdim': xdim, 'ydim': ydim}
    with open(output_filepath + '/rf_DHT{}.pickle'.format(centered * '_centered'), 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def extract_V1_rf_Ringach(input_filepath, output_filepath, centered=True):
    """
    Loads the V1 receptive fields dataset given by Dario Ringach in the paper. In the data, 
    the RF sizes are different from 32 x 32, 64 x 64, and 128 x 128. We resize them to 
    32 x 32 for analysis. 
    
    Ref: 'Spatial structure and symmetry of simple-cell receptive fields in 
    macaque primary visual cortex'
    
    Parameters
    ----------
    input_filepath: str
        Location of V1 rfs from Dario Ringach

    output_filepath: str
        output location

    centered: Bool
        Move the receptive fields to the center of the image 

    Returns
    -------
    output_dict: dict
        Dictionary with centered rf & snr for every cell, xdim and ydim of
        the rf 
    """

    data_dir = input_filepath + '/rf_Ringach.mat'

    # load data
    RF_ringach = loadmat(data_dir)['rf']
    num_cells = RF_ringach.shape[1]
    xdim, ydim = 32, 32  # reshape rf to 32 x 32 images
    processed_RF = np.zeros((num_cells, xdim * ydim)) 

    for cell in range(num_cells):
        cell_rf = RF_ringach[0][cell][0]

        # resize to 32 x 32
        resize_factor = int(cell_rf.shape[0] / xdim)
        cell_rf = downscale_local_mean(cell_rf, (resize_factor, resize_factor))

        # center
        if centered == True:
            centered_rf = center_receptive_field(cell_rf, xdim, ydim)
            processed_RF[cell] = centered_rf.flatten()

        elif centered == False:
            processed_RF[cell] = cell_rf.flatten()

    # save into an output dict and pickle dump
    output_dict = {'rf': processed_RF, 'xdim': xdim, 'ydim': ydim}
    with open(output_filepath + '/rf_Ringach{}.pickle'.format(centered * '_centered'), 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def download_MNIST(output_file_path):
    """
    Download the MNIST dataset using pytorch. 

    Parameters
    ----------
    output_filepath: str
        output location
    """
    datasets.MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
]
    _ = datasets.MNIST(output_file_path + '/', train=True, download=True)
    _ = datasets.MNIST(output_file_path + '/', train=False, download=True)


def download_KMNIST(output_file_path):
    """
    Download the KMNIST dataset using pytorch. 

    Parameters
    ----------
    output_filepath: str
        output location
    """

    _ = datasets.KMNIST(output_file_path + '/', train=True, download=True)
    _ = datasets.KMNIST(output_file_path + '/', train=False, download=True)



