# argparse stuff
import pywt
import argparse
import sys
sys.path.append("../")

def add_colorbar_args(parser):
    '''
    Add arguments to the parser that are required for 
    colorbar_live_reconst function in figure_library.py

    Parameters
    ----------
    parser : ArgumentParser
        parser object used to hold argument information
    '''

    # theres a lot of these -- use this function instead of manually typing all
    wavelist = pywt.wavelist()
    # get image infile
    
    parser.add_argument('-observation', choices=['pixel', 'V1', 'gaussian'], action='store', metavar='OBS', help='[Colorbar Figure] : observation type to use when sampling', required=False, nargs=1)
    parser.add_argument('-mode', choices=['color', 'black'], action='store', metavar='COLOR_MODE', help='[Colorbar Figure] : color mode of reconstruction', required=False, nargs=1)
    # add hyperparams REQUIRED for dwt ONLY
    parser.add_argument('-dwt_type', choices=wavelist, action='store', metavar='DWT_TYPE', help='[Colorbar Figure] : dwt type', required=False, nargs=1)
    parser.add_argument('-level', choices=['1', '2', '3', '4'], action='store', metavar='LEVEL', help='[Colorbar Figure] : level', required=False, nargs=1)
    # add hyperparams REQUIRED for V1 ONLY
    parser.add_argument('-cell_size', action='store', metavar='CELL_SIZE', help='[Colorbar Figure] : cell size', required=False, nargs=1)
    parser.add_argument('-sparse_freq', action='store', metavar='FREQ', help='[Colorbar Figure] : sparse frequency', required=False, nargs=1)
    # add hyperparams that are used for both dct and dwt
    parser.add_argument('-alpha', action='store', metavar="ALPHA", help='[Colorbar Figure] : alpha values to use', required=False, nargs=1)
    parser.add_argument('-num_cells', action='store', metavar='NUM_CELLS', help='[Colorbar Figure] : Method you would like to use for reconstruction', required=False, nargs=1)
    

def eval_colorbar_args(args, parser):
    '''
    Evaluate the colorbar arguments to 
    ensure that they are sufficient for plotting.
    
    Parameters
    ----------
    args : NameSpace
        Contains each arg and its assigned value.

    parser : ArgumentParser
        Parser object used to give error if needed.

    Returns
    -------
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
        Level of signal frequencies for dwt.
        Should be an integer in [1, 4].
        
    alpha : float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents.

    num_cells : int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    cell_size : int
        Determines field size of opened and closed blob of data. 
        Affect the data training

    sparse_freq : int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training
    '''
    method = args.method[0] if args.method is not None else None
    img_name = args.img_name[0] if args.img_name is not None else None
    observation = args.observation[0] if args.observation is not None else None
    mode = args.mode[0] if args.mode is not None else None
    num_cells = eval(args.num_cells[0]) if args.num_cells is not None else None
    if None in [method, img_name, observation, mode, num_cells]:
        parser.error('[Colorbar Figure] : at least method, img_name, observation, mode, num_cells required for colorbar figure')
    # deal with missing or unneccessary command line args
    if method == "dwt" and (args.dwt_type is None or args.level is None):
        parser.error('[Colorbar Figure] : dwt method requires -dwt_type and -level.')
    elif method == "dct" and (args.dwt_type is not None or args.level is not None):
        parser.error('[Colorbar Figure] : dct method does not use -dwt_type and -level.')
    if observation.lower() == "v1" and (args.cell_size is None or args.sparse_freq is None):
        parser.error('[Colorbar Figure] : V1 observation requires cell size and sparse freq.')
    elif observation.lower() != "v1" and (args.cell_size is not None or args.sparse_freq is not None):
        parser.error('[Colorbar Figure] : Cell size and sparse freq params are only required for V1 observation.')
    dwt_type = eval(args.dwt_type[0]) if args.dwt_type is not None else None
    level = eval(args.level[0]) if args.level is not None else None
    alpha = eval(args.alpha[0]) if args.alpha is not None else None
    cell_size = eval(args.cell_size[0]) if args.cell_size is not None else None
    sparse_freq = eval(args.sparse_freq[0]) if args.sparse_freq is not None else None
    return method, img_name, observation, mode, dwt_type, level, alpha, num_cells, cell_size, sparse_freq


def add_num_cell_args(parser):
    '''
    Add arguments to the parser that are required for 
    the num_cell_error function in figure_library.py

    Parameters
    ----------
    parser : ArgumentParser
        parser object from argparse used to add arguments
    '''
    
    parser.add_argument('-pixel_file', action='store', metavar='FILE', help='[Num Cell Figure] : file to read pixel data from', required=False, nargs=1)
    parser.add_argument('-gaussian_file', action='store', metavar='FILE', help='[Num Cell Figure] : file to read gaussian data from', required=False, nargs=1)
    parser.add_argument('-v1_file', action='store', metavar='FILE', help='[Num Cell Figure] : file to read V1 data from', required=False, nargs=1)
    parser.add_argument('-data_grab', action='store_true', help='[Num Cell Figure] : auto grab data when argument is present', required=False)
    parser.add_argument('-save', action='store_true', help='[Num Cell Figure] : save into specified path when argument is present', required=False)

def eval_num_cell_args(args, parser):
    '''
    Evaluate the args to ensure that they are sufficient for 
    generating num cell error plot.
    
    Parameters
    ----------
    args : NameSpace
        Contains each arg and its assigned value.

    parser : ArgumentParser
        Parser object used to give error if needed.

    Returns
    -------
    img_name : String
        The name of the image that was reconstructed

    method : String
        The basis being used for reconstruction [dct, dwt]

    pixel: String
       csv file containing pixel reconstruction data

    gaussian : String
       csv file containing gaussian reconstruction data

    v1 : String
       csv file containing V1 reconstruction data

    data_grab : boolean
       TODO: convert boolean to 'auto' or 'manual'

    save : boolean
       Indicates if generated plot should be saved to a file.

    '''

    img_name = args.img_name[0] if args.img_name is not None else None
    method = args.method[0] if args.method is not None else None
    pixel = args.pixel_file[0] if args.pixel_file is not None else None
    gaussian = args.gaussian_file[0] if args.gaussian_file is not None else None
    v1 = args.v1_file[0] if args.v1_file is not None else None
    data_grab = args.data_grab# if args.data_grab is not None else None
    save = args.save# if args.save is not None else None
    if None in [method, img_name, pixel, gaussian, v1]:
        parser.error('[Num Cell Figure] : at least method, img_name, pixel_file, gaussian_file, V1_file required for num cell error figure')
    return img_name, method, pixel, gaussian, v1, data_grab, save

def add_generic_figure_args(parser):
    ''' 
    Add arguments that are used in every plotting function
    
    Parameters
    ----------
    parser : ArgumentParser
        Parser object used to hold argument information
    '''
    # add figtype -- this is the only required argparse arg, determine which others should be there based on figtype
    parser.add_argument('-fig_type', choices=['colorbar', 'num_cell'], action='store', metavar='FIGTYPE', help='[Colorbar and Num Cell Figure] : type of figure to generate', required=True, nargs=1)
    # add arguments used by both num cell and colorbar
    parser.add_argument('-img_name', action='store', metavar='IMG_NAME', help='[Colorbar and Num Cell Figure] : filename of image to be reconstructed', required=False, nargs=1)
    parser.add_argument('-method', choices=['dct', 'dwt'], action='store', metavar='METHOD', help='[Colorbar and Num Cell Figure] : Method to use for reconstruction', required=False, nargs=1)


def parse_figure_args():
    '''
    Parse the command line args for the figure library
    
    Returns
    -------
    figure type : String
        Desired type of figure to plot 'colorbar' or 'num_cell'

    params : List
        Returns the appropriate parameters to run for a given figure type.
    '''
    parser = argparse.ArgumentParser(description='Generate a figure of your choosing.')
    add_generic_figure_args(parser)
    add_colorbar_args(parser)
    add_num_cell_args(parser)
    args = parser.parse_args()
    fig_type = args.fig_type[0]
    if fig_type == 'colorbar':
        params = eval_colorbar_args(args, parser)
    elif fig_type == 'num_cell':
        params = eval_num_cell_args(args, parser)

    return fig_type, params




def parse_sweep_args():
    '''
    Parse the command line args for the hyperparam sweep filter
    
    Returns
    ----------
    method : String
        Method of reconstruction ('dwt' or 'dct').
    
    img : String
        Name of image to reconstruct (e.g. 'tree_part1.jpg').

    observation : String
        Method of observation (e.g. pixel, gaussian, v1).
    
    mode : String
        Desired mode to reconstruct image.
        (e.g. 'Color' for RGB, 'Black' for greyscaled images).

    dwt_type : String
        Type of dwt method to be used.
        See pywt.wavelist() for all possible dwt types.
        
    lv : List of int
        List of one or more integers in [1, 4].
        
    alpha_list : List of float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents.

    num_cell : List of int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    cell_size : List of int
        Determines field size of opened and closed blob of data. 
        Affect the data training.

    sparse_freq : List of int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.
    '''

    parser = argparse.ArgumentParser(description='Create a hyperparameter sweep')
    add_sweep_args(parser)
    args = parser.parse_args()
    method, img_name, observation, mode, dwt_type, level, alpha_list, num_cells, cell_size, sparse_freq = eval_sweep_args(args, parser)
    return method, img_name, observation, mode, dwt_type, level, alpha_list, num_cells, cell_size, sparse_freq

def add_sweep_args(parser):
    '''
    Add arguments to the parser that are required for the hyperparameter sweep

    Parameters
    ----------
    parser : ArgumentParser
        parser object used to hold argument information
    '''
    # theres a lot of these -- use this function instead of manually typing all
    wavelist = pywt.wavelist()
    
    # get image infile
    parser.add_argument('-img_name', action='store', metavar='IMG_NAME', help='filename of image to be reconstructed', required=True, nargs=1)
    # add standard params
    parser.add_argument('-method', choices=['dct', 'dwt'], action='store', metavar='METHOD', help='Method you would like to use for reconstruction', required=True, nargs=1)
    parser.add_argument('-observation', choices=['pixel', 'V1', 'gaussian'], action='store', metavar='OBSERVATION', help='observation type to use when sampling', required=True, nargs=1)
    parser.add_argument('-mode', choices=['color', 'black'], action='store', metavar='COLOR_MODE', help='color mode of reconstruction', required=True, nargs=1)
    # add hyperparams REQUIRED for dwt ONLY
    parser.add_argument('-dwt_type', choices=wavelist, action='store', metavar='DWT_TYPE', help='dwt type', required=False, nargs=1)
    parser.add_argument('-level', choices=['1', '2', '3', '4'], action='store', metavar='LEVEL', help='level of signal frequency for dwt', required=False, nargs="+")
    # add hyperparams REQUIRED for v1 only
    parser.add_argument('-cell_size', action='store', metavar='CELL_SIZE', help='cell size', required=False, nargs="+")
    parser.add_argument('-sparse_freq', action='store', metavar='SPARSE_FREQUENCY', help='sparse frequency', required=False, nargs="+")
    # add hyperparams that are used for both dct and dwt
    parser.add_argument('-alpha_list', action='store', metavar="ALPHAS", help='alpha values to use', required=True, nargs="+")
    parser.add_argument('-num_cells', action='store', metavar='NUM_CELLS', help='Method you would like to use for reconstruction', required=True, nargs="+")

def eval_sweep_args(args, parser):
    '''
    Evaluate the colorbar arguments to 
    ensure that they are sufficient for plotting.
    
    Parameters
    ----------
    args : NameSpace
        Contains each arg and its assigned value.

    parser : ArgumentParser
        Parser object used to give error if needed.

    Returns
    -------
    method : String
        Method of reconstruction ('dwt' or 'dct').
    
    img : String
        Name of image to reconstruct (e.g. 'tree_part1.jpg').

    observation : String
        Method of observation (e.g. pixel, gaussian, v1).
    
    mode : String
        Desired mode to reconstruct image 
        (e.g. 'Color' for RGB, 'Black' for greyscaled images).

    dwt_type : String
        Type of dwt method to be used
        See pywt.wavelist() for all possible dwt types.
        
    lv : List of int
        List of one or more integers in [1, 4].
        
    alpha_list : List of float
        Penalty for fitting data onto LASSO function to 
        search for significant coefficents.

    num_cell : List of int
        Number of blobs that will be used to be 
        determining which pixels to grab and use.
    
    cell_size : List of int
        Determines field size of opened and closed blob of data. 
        Affect the data training.

    sparse_freq : List of int
        Determines filed frequency on how frequently 
        opened and closed area would appear. 
        Affect the data training.
    '''
    
    #args = parser.parse_args()
    method = args.method[0]
    img_name = args.img_name[0]
    observation = args.observation[0]
    mode = args.mode[0]
    # deal with missing or unneccessary command line args
    if method == "dwt" and (args.dwt_type is None or args.level is None):
        parser.error('dwt method requires -dwt_type and -level.')
    elif method == "dct" and (args.dwt_type is not None or args.level is not None):
        parser.error('dct method does not use -dwt_type and -level.')
    if observation.lower() == "v1" and (args.cell_size is None or args.sparse_freq is None):
        parser.error('v1 observation requires cell size and sparse freq.')
    elif observation.lower() != "v1" and (args.cell_size is not None or args.sparse_freq is not None):
        parser.error('Cell size and sparse freq params are only required for V1 observation.')
    dwt_type = args.dwt_type
    level = [eval(i) for i in args.level] if args.level is not None else None
    alpha_list = [eval(i) for i in args.alpha_list]
    num_cells = [eval(i) for i in args.num_cells]
    cell_size = [eval(i) for i in args.cell_size] if args.cell_size is not None else None
    sparse_freq = [eval(i) for i in args.sparse_freq] if args.sparse_freq is not None else None

    return method, img_name, observation, mode, dwt_type, level, alpha_list, num_cells, cell_size, sparse_freq