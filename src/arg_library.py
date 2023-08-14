# argparse stuff
import pywt
import argparse
import sys
sys.path.append("../")

def add_colorbar_args(parser):
    '''
    Add arguments to the parser that are required for the colorbar_live_reconst function in figure_library.py

    Parameters
    ----------
    parser : ???
        parser object from argparse used to add arguments

    '''

    # theres a lot of these -- use this function instead of manually typing all
    wavelist = pywt.wavelist()
    # get image infile
    
    parser.add_argument('-img_name', action='store', metavar='IMG_NAME', help='[Colorbar and Num Cell Figure] : filename of image to be reconstructed', required=False, nargs=1)
    # add standard params
    parser.add_argument('-method', choices=['dct', 'dwt'], action='store', metavar='METHOD', help='[Colorbar and Num Cell Figure] : Method to use for reconstruction', required=False, nargs=1)
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
    Add arguments to the parser that are required for the num_cell_error function in figure_library.py

    Parameters
    ----------
    parser : ???
        parser object from argparse used to add arguments

    '''
    
    parser.add_argument('-pixel_file', action='store', metavar='FILE', help='[Num Cell Figure] : file to read pixel data from', required=False, nargs=1)
    parser.add_argument('-gaussian_file', action='store', metavar='FILE', help='[Num Cell Figure] : file to read gaussian data from', required=False, nargs=1)
    parser.add_argument('-v1_file', action='store', metavar='FILE', help='[Num Cell Figure] : file to read V1 data from', required=False, nargs=1)
    parser.add_argument('-data_grab', action='store_true', help='[Num Cell Figure] : auto grab data when argument is present', required=False)
    parser.add_argument('-save', action='store_true', help='[Num Cell Figure] : save into specified path when argument is present', required=False)

def eval_num_cell_args(args, parser):
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
    
    # add figtype -- this is the only required argparse arg, determine which others should be there based on figtype
    parser.add_argument('-fig_type', choices=['colorbar', 'num_cell'], action='store', metavar='FIGTYPE', help='[Colorbar and Num Cell Figure] : type of figure to generate', required=True, nargs=1)

    add_colorbar_args(parser)
    add_num_cell_args(parser)
    args = parser.parse_args()
    fig_type = args.fig_type[0]
    if fig_type == 'colorbar':
        params = eval_colorbar_args(args, parser)
    elif fig_type == 'num_cell':
        params = eval_num_cell_args(args, parser)

    return fig_type, params

