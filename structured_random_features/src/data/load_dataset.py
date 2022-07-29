
import os.path as path

import numpy as np
import scipy
import sklearn.utils as sk_utils
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision import datasets, transforms

data_dir = path.abspath(path.join(__file__, "../../../"))
data_dir += '/data/processed/'

def load_sensilla_sta():
    """
    Construct spike triggered averages (STA) from sensilla mechanosensors. We are
     using data provided by Brandon Pratt.
     Neural evidence supports a dual sensory-motor role for insect wings 
     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5597827/

     Returns
     -------
     STA: (array-like) of shape (95, 1600)
        STA of 95 mechanosensory cells

    sample_rate: float
        Sampling rate of the neuron in the experiment

    sta_time_window: float
        Length of time for the STA (seconds) 
    """
    data_path = data_dir + 'STA_sensilla.pickle'
    assert path.exists(data_path), 'STA data doesnt exist in data/processed. Check src/data/processing.py'
    outputs = np.load(data_path, allow_pickle=True)
    return outputs['STA'], outputs['sample_rate'], outputs['sta_time_window']


def load_V1_whitenoise(centered=True):
    '''
    Loads the V1 receptive fields dataset given by Marius Pachitariu. 44k neurons were
    shown white noise repeatedly and their RFs were calculated. In the data, 
    the RF sizes are 14 x 36. We normalize the RFs and center them for analysis.
    
    Parameters
    ----------
    centered: bool, default=True
        If True, centers the receptive fields to (7, 18) pixel coordinate using
        the image's center of mass.
    
    Returns
    -------
    rf: (array-like) of shape (45026, 504)
        Receptive fieds of ~45k neurons.
        
    snr: (array-like) of shape (45026, 1)
        SNR of receptive fields

    (xdim, ydim): tuple of shape (2, 1)
        Dimension of the receptive fields
        
    '''
    data_path = data_dir + 'rf_whitenoise{}.pickle'.format('_centered' * centered)
    assert path.exists(data_path), '{} RF data doesnt exist in data/processed. Check src/data/processing.py'.format('Centered' * centered)
    outputs = np.load(data_path, allow_pickle=True)
    return outputs['rf'], outputs['snr'], (outputs['xdim'], outputs['ydim'])


def load_V1_DHT(centered=True):
    '''
    Loads the V1 receptive fields dataset given by Marius Pachitariu. 4k neurons were
    shown hartley basis functions repeatedly and their RFs were calculated. In the data, 
    the RF sizes are 30 x 80. We normalize the RFs and center them for analysis.
    
    Parameters
    ----------
    centered: bool, default=True
        If True, centers the receptive fields to (15, 40) pixel coordinate using
        the image's center of mass.
    Returns
    -------
    rf: (array-like) of shape (4337, 2400)
        Receptive fieds of ~4k neurons.
    
    snr: (array-like) of shape (4337, 1)
        SNR of receptive fields

    (xdim, ydim): tuple of shape (2, 1)
        Dimension of the receptive fields
        
    '''
    data_path = data_dir + 'rf_DHT{}.pickle'.format('_centered' * centered)
    assert path.exists(data_path), '{} RF data doesnt exist in data/processed. Check src/data/processing.py'.format('Centered' * centered)
    outputs = np.load(data_path, allow_pickle=True)
    return outputs['rf'], outputs['snr'], (outputs['xdim'], outputs['ydim'])


def load_V1_natural_images(centered=True):
    '''
    Loads the V1 receptive fields dataset given by Marius Pachitariu. 70k neurons were
    shown 5000 naturalistic images in random order in 3 trials. In the data, 
    the RF sizes are 24 x 27. We normalize them and center them for analysis.
    
    Parameters
    ----------
    centered: bool, default=True
        If True, centers the receptive fields to (12, 13) pixel coordinate using
        the image's center of mass.
    
    Returns
    -------
    rf: (array-like) of shape (3, 69957, 24 * 27)
        Receptive fieds of ~70k neurons from 3 trials.
        
    snr: (array-like) of shape (69957, 1)
        SNR of receptive fields
    
    (xdim, ydim): tuple of shape (2, 1)
        Dimension of the receptive fields
        
    '''
    data_path = data_dir + 'rf_natural_images{}.pickle'.format('_centered' * centered)
    assert path.exists(data_path), '{} RF data doesnt exist in data/processed. Check src/data/processing.py'.format('Centered' * centered)
    outputs = np.load(data_path, allow_pickle=True)
    return outputs['rf'], outputs['snr'], (outputs['xdim'], outputs['ydim'])


def load_V1_Ringach(centered=True):
    """
    Loads the V1 receptive fields dataset given by Dario Ringach in the paper. In the data, 
    the RF sizes are different from 32 x 32, 64 x 64, and 128 x 128. We resize them to 
    32 x 32 for analysis. 
    
    Ref: 'Spatial structure and symmetry of simple-cell receptive fields in 
    macaque primary visual cortex'
    
    Parameters
    ----------
    data_dir: string, default='./data/V1_data_Ringach/' 
        Path to the receptive field data folder
    
    centered: bool, default=True
        If True, centers the receptive fields to (16, 16) pixel coordinate.
    
    normalized: bool, default=True
        If True, the mean of RF values is 0 and std dev is 1. 
    
    Returns
    -------
    processed_RF:  (array-like) of shape (250, 1024)
        Receptive fields of 250 neurons
    """
    data_path = data_dir + 'rf_Ringach{}.pickle'.format('_centered' * centered)
    assert path.exists(data_path), '{} RF data doesnt exist in data/processed. Check src/data/processing.py'.format('Centered' * centered)
    outputs = np.load(data_path, allow_pickle=True)
    return outputs['rf'], (outputs['xdim'], outputs['ydim'])


def get_dataloader(train_set, test_set, train_batch_size, test_batch_size=1024, val_batch_size=1024, train_percentage=0.8, seed=None):
    """
    Given training set and test set, splits the training set into a validation set using the train percentage. Then, returns the dataloaders for train, test, and validation data.

    Parameters
    ----------

    train_set: torch.utils.data.TensorDataset object
        training set

    test_set: torch.utils.data.TensorDataset object
        test set

    train_batch_size: int
        batch size of the training set

    test_batch_size: int, default=1024
        batch size of the test set

    val_batch_size: int, default=1024
        batch size of the validation set

    train_percentage: float, default=0.8
        percentage of train set used for training
        Rest is used for validation
        
    """
    if seed is not None:
        torch.manual_seed(seed)

    # generate validation set
    train_subset_size = int(len(train_set) * train_percentage)
    train_subset, val_set = random_split(train_set, [train_subset_size, len(train_set) - train_subset_size])

    # put into dataloaders
    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=True)
    return train_loader, val_loader, test_loader



def generate_frequency_detection(num_samples, sampling_rate, freq, duration, snr, seed=None):
    """
    Generate frequency detection task. The positive examples are pure
    sinusoids with additive gaussian noise and the negative examples are 
    white noise. The examples are generated using the DFT matrix.
    
    The squared L2 norm of the generated examples over the interval
    duration is 1. 
    \int^duration_0 x^2 dx = 1

    Parameters
    ----------

    num_samples : int
        Number of total examples

    sampling_rate : int
        Sampling rate of the signal in Hz
    
    freq : int
        Frequency of the signal in Hz

    duration: float
        Length of the signal in seconds.

    snr : float, 0 <= a <= 1
        Determines the SNR of the signal.
        SNR = a ** 2 / (1 - a ** 2)

    seed : int
        Random state of the generated examples

    Returns
    -------
    X : (array-like) of shape (num_samples, num_features)
        Every row corresponds to an example with num_features components.
        num_features =  sampling_rate * duration

    y : (array-like) of shape (num_samples,)
        Target label (0/1) for every example. 
    """

    np.random.seed(seed)
    
    N = int(sampling_rate * duration)
    noise_amplitude = np.sqrt(1 - snr ** 2)

    # dft matrix
    A = scipy.linalg.dft(N, scale='sqrtn')
    idx = int(freq * duration) # row of DFT matrix that corresponds to the frequency

    # positive examples
    n_pos = int(num_samples / 2)
    c = np.zeros((N, n_pos), dtype='complex')
    rand = np.random.normal(loc=0, scale=1, size=(n_pos, 2)).view(complex).flatten()
    rand /= np.abs(rand)
    c[idx] = rand
    X_pos = np.sqrt(2 / duration) * snr * (A @ c).T.real

    # noise for positive egs
    rand = np.random.normal(loc=0, scale=1, size=(N, n_pos, 2)).view(complex).squeeze(axis=2)
    rand /= np.abs(rand)
    rand[idx] = 0. # don't add noise to the signal component
    noise = np.sqrt(2 / ((N-1) * duration)) * noise_amplitude * (A @ rand).T.real
    X_pos += noise

    # negative egs
    n_neg = int(num_samples / 2)
    c = np.random.normal(loc=0, scale=1, size=(N, n_neg, 2)).view(complex).squeeze(axis=2)
    c /= np.abs(c)
    X_neg = np.sqrt(2 / (N * duration)) * (A @ c).T.real

    # concatenate and shuffle
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_pos), np.zeros(n_neg)))
    X, y = sk_utils.shuffle(X, y)

    return X, y

def load_frequency_detection(num_samples, sampling_rate, freq, duration, snr, train_batch_size=1024, train_percentage=0.8, seed=None):
    """
    Generate train, validation, and test dataloaders for the frequency detection task. Here, num_samples will be divided into train and validation set according to train percentage.
    - The test set is hardcoded to have 10000 points. 
    - The batch_size for validation and test set is hardcoded to 1024.

    Parameters
    ----------

    n_samples : int
        Number of total examples

    sampling_rate : int
        Sampling rate of the signal in Hz
    
    freq : int
        Frequency of the signal in Hz

    duration: float
        Length of the signal in seconds.

    snr : float, 0 <= a <= 1
        Determines the SNR of the signal.
        SNR = a ** 2 / (1 - a ** 2)

    train_batch_size : int, default=1024
        Batch size of the training set

    train_percentage: float, default=0.8
        Percentage of num_samples used for training. 
        Rest will be used for validation.
    
    seed : int
        Seed for the generated examples

    Returns
    -------

    train_loader: torch DataLoader object
        Dataloader for the training data

    val_loader: torch DataLoader object
        DataLoader for the validation data

    test_loader: torch DataLoader object
        DataLoader for the test data

    """

    train_data, train_labels = generate_frequency_detection(num_samples, sampling_rate, freq, duration, snr, seed=seed)
    test_data, test_labels = generate_frequency_detection(10000, sampling_rate, freq, duration, snr, seed=seed)

    # convert to pytorch tensor and then to tensor dataset format
    train_set = TensorDataset(torch.Tensor(train_data), torch.LongTensor(train_labels))
    test_set = TensorDataset(torch.Tensor(test_data), torch.LongTensor(test_labels))

    # get dataloaders 
    train_loader, val_loader, test_loader = get_dataloader(train_set, test_set, train_batch_size=train_batch_size, test_batch_size=1024, val_batch_size=1024, train_percentage=train_percentage, seed=seed)

    return train_loader, val_loader, test_loader


def generate_frequency_XOR(num_samples, sampling_rate, freq1, freq2, duration, snr, seed=None, shuffle=True):
    """
    Generates a frequency XOR task. The positive eg are single
    frequency sinusoids (2 frequencies) with additive gaussian noise. 
    The negative eg are mixed sinusoids or white noise.
    
    The squared L2 norm of the generated examples over the interval
    duration is 1. 
    \int^duration_0 x^2 dx = 1

    Parameters
    ----------

    num_samples : int
        Number of total examples

    sampling_rate : int
        Sampling rate of the signal in Hz
    
    freq1 : int
        Frequency 1 of the signal in Hz

    freq2 : int
        Frequency 2 of the signal in Hz

    duration: float
        Length of the signal in seconds.

    snr : float, 0 <= a <= 1
        Determines the SNR of the signal.
        SNR = a ** 2 / (1 - a ** 2)

    seed : int
        Random state of the generated examples
    
    shuffle : Bool, default=True
        Shuffle data if True.
    
    Returns
    -------

    X : (array-like) of shape (num_samples, n_features)
        Every row corresponds to an example with n_features components.
        n_features = fs * duration

    y : (array-like) of shape (num_samples,)
        Target label (0/1) for every example. 
    """

    np.random.seed(seed)

    N = int(sampling_rate * duration)
    noise_amplitude = np.sqrt(1 - snr ** 2)

    #dft matrix
    A = scipy.linalg.dft(N, scale='sqrtn')
    idx1 = int(duration * freq1) # row of DFT matrix that corresponds to the frequency
    idx2 = int(duration * freq2)

    # positive examples
    n_pos = int(num_samples/2)
    c = np.zeros((N, n_pos), dtype='complex')
    rand = np.random.normal(loc=0, scale=1, size=(int(n_pos/ 2), 2)).view(complex).flatten()
    rand /= np.abs(rand)
    c[idx1, :int(n_pos/2)] = rand

    rand = np.random.normal(loc=0, scale=1, size=(int(n_pos/ 2), 2)).view(complex).flatten()
    rand /= np.abs(rand)
    c[idx2, int(n_pos/2):] = rand
    X_pos = np.sqrt(2 / duration) * snr * (A @ c).T.real

    # noise for positive egs
    rand = np.random.normal(loc=0, scale=1, size=(N, n_pos, 2)).view(complex).squeeze(axis=2)
    rand /= np.abs(rand)
    rand[idx1, :int(n_pos/2)] = 0. # don't add noise for signal frequency
    rand[idx2, int(n_pos/2):] = 0.
    noise = np.sqrt(2 / ((N-1) * duration)) * noise_amplitude * (A @ rand).T.real
    X_pos += noise

    # negative egs
    n_neg = int(num_samples/2)

    # mixed egs
    c = np.zeros((N, int(n_neg/2)), dtype='complex')
    rand = np.random.normal(loc=0, scale=1, size=(1, int(n_neg/2), 2)).view(complex).squeeze(axis=2)
    rand /= np.abs(rand)
    c[[idx1, idx2]] = rand
    X_mixed = np.sqrt(1 / duration) * snr * (A @ c).T.real

    # noise for mixed egs
    rand = np.random.normal(loc=0, scale=1, size=(N, int(n_neg/2), 2)).view(complex).squeeze(axis=2)
    rand /= np.abs(rand)
    rand[[idx1, idx2]] = 0.
    noise = np.sqrt(2 / ((N-2) * duration)) * noise_amplitude * (A @ rand).T.real
    X_mixed += noise

    # noise as negative egs
    c = np.random.normal(loc=0, scale=1, size=(N, int(n_neg/2), 2)).view(complex).squeeze(axis=2)
    c /= np.abs(c)
    X_noise = np.sqrt(2 / (N * duration)) * (A @ c).T.real
    
    X_neg = np.row_stack((X_mixed, X_noise))
    
    # concatenate and shuffle
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(n_pos), np.zeros(n_neg)))
    if shuffle is True:
        X, y = sk_utils.shuffle(X, y)
    return X, y


def load_frequency_XOR(num_samples, sampling_rate, freq1, freq2, duration, snr, train_batch_size=1024, train_percentage=0.8, seed=None):
    """
    Generate train, validation, and test dataloaders for the frequency XOR task. Here, num_samples will be divided into train and validation set according to train percentage.
    - The test set is hardcoded to have 10000 points. 
    - The batch_size for validation and test set is hardcoded to 1024.

    Parameters
    ----------

    num_samples : int
        Number of total examples

    sampling_rate : int
        Sampling rate of the signal in Hz
    
    freq1 : int
        Frequency 1 of the signal in Hz

    freq2 : int
        Frequency 2 of the signal in Hz

    duration: float
        Length of the signal in seconds.

    snr : float, 0 <= a <= 1
        Determines the SNR of the signal.
        SNR = a ** 2 / (1 - a ** 2)

    seed : int
        Random state of the generated examples

    train_batch_size : int, default=1024
        Batch size of the training set

    train_percentage: float, default=0.8
        Percentage of num_samples used for training. 
        Rest will be used for validation.
    
    seed : int
        Seed for the generated examples

    Returns
    -------

    train_loader: torch DataLoader object
        Dataloader for the training data

    val_loader: torch DataLoader object
        DataLoader for the validation data

    test_loader: torch DataLoader object
        DataLoader for the test data

    """
    train_data, train_labels = generate_frequency_XOR(num_samples, sampling_rate, freq1, freq2, duration, snr, seed=seed, shuffle=True)
    test_data, test_labels = generate_frequency_XOR(10000, sampling_rate, freq1, freq2, duration, snr, seed=seed, shuffle=True)

    # convert to pytorch tensor and then to tensor dataset format
    train_set = TensorDataset(torch.Tensor(train_data), torch.LongTensor(train_labels))
    test_set = TensorDataset(torch.Tensor(test_data), torch.LongTensor(test_labels))

    # get dataloaders 
    train_loader, val_loader, test_loader = get_dataloader(train_set, test_set, train_batch_size=train_batch_size, test_batch_size=1024, val_batch_size=1024, train_percentage=train_percentage, seed=seed)

    return train_loader, val_loader, test_loader


def load_mnist(train_batch_size=1024, train_percentage=0.8, seed=None):
    """
    Generate train, test, and val loader for the MNIST dataset.
    - The batch_size for validation and test set is hardcoded to 1024.

    Parameters
    ----------

    train_batch_size : int, default=1024
        Batch size of the training set

    train_percentage: float, default=0.8
        Percentage of num_samples used for training. 
        Rest will be used for validation.
    
    seed : int
        Seed used when randomly splitting train data into train and test set

    Returns
    -------

    train_loader: torch DataLoader object
        Dataloader for the training data

    val_loader: torch DataLoader object
        DataLoader for the validation data

    test_loader: torch DataLoader object
        DataLoader for the test data
    """

    # load data and apply normalization
    transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=False, transform=transform)

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloader(train_set, test_set, train_batch_size=train_batch_size, test_batch_size=1024, val_batch_size=1024, train_percentage=train_percentage, seed=seed)
    return train_loader, val_loader, test_loader


def load_kmnist(train_batch_size=1024, train_percentage=0.8, seed=None):
    """
    Generate train, test, and val loader for the KMNIST dataset.
    - The batch_size for validation and test set is hardcoded to 1024.

    Parameters
    ----------

    train_batch_size : int, default=1024
        Batch size of the training set

    train_percentage: float, default=0.8
        Percentage of num_samples used for training. 
        Rest will be used for validation.
    
    seed : int
        Seed used when randomly splitting train data into train and test set

    Returns
    -------

    train_loader: torch DataLoader object
        Dataloader for the training data

    val_loader: torch DataLoader object
        DataLoader for the validation data

    test_loader: torch DataLoader object
        DataLoader for the test data
    """

    # load data and apply normalization
    transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.KMNIST(data_dir, train=True, download=False, transform=transform)
    test_set = datasets.KMNIST(data_dir, train=False, download=False, transform=transform)

    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloader(train_set, test_set, train_batch_size=train_batch_size, test_batch_size=1024, val_batch_size=1024, train_percentage=train_percentage, seed=seed)
    return train_loader, val_loader, test_loader

    
