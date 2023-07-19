import numpy as np
import numpy.linalg as la
import sys
sys.path.append("../")
import matplotlib.pyplot as plt

# Package for importing image representation
from PIL import Image, ImageOps

from src.compress_sensing_library import *
from src.utility_library import *
import pandas as pd
import itertools
import dask
from dask.distributed import Client, progress
import time
import os.path

def usage():
    print(
        '''Usage
        arg[1] = img_name ex)tree_part1.jpg
        arg[2] = observation_type ex)pixel, V1, gaussian
        arg[3] = mode ex) color/black
        arg[4] = alpha_list ex) [0.001, 0.01, 0.1]
        arg[5] = num_cell list ex) [50, 100, 200, 500]
        arg[6] = cell_size list ex) [1, 2, 4, 8, 10]
        arg[7] = sparse_freq ex) [1, 2, 4, 8, 10]
        '''
    )
    sys.exit(0)

def parse_input(i, argv):
    default = {
        1 : 'tree_part1.jpg',
        2 : 'pixel',
        3 : 'black',
        4 : list(np.logspace(-2, 2, 5)),
        5 : [50, 100, 200, 500],
        6 : list(np.arange(1, 11, 1)),
        7 : list(np.arange(1, 11, 1))
        }
    
    ## Checking statement if default is formed correctly
    #print("In parse_input, default {i} is {default}".format(i = i, default = default.get(i)))
    
    if (i != 1 and (argv[i].lower() in ['n', '-n', 'none', '-none'])):
        return default.get(i)
    
    if (i == 1):
        assert argv[i] != 'n'
        return argv[i]
    elif (i == 2):
        return argv[i]
    elif (i == 3):
        if argv[i].lower() in ['-c', 'color']:
            return "color"
        return 'black'
    elif (i == 4):
        lst = list(argv[i].strip('[]').replace(" ", "").split(','))
        print(lst)
        lst = list(map(lambda x: float(x) , lst))
    elif (i == 5) :
        lst = lst = list(argv[i].strip('[]').replace(" ", "").split(','))
        if (lst[0].isdigit):
            lst = list(map(lambda x: int(x) , lst))
        else:
            lst = list(map(lambda x: float(x) , lst))
    else:
        lst = list(argv[i].strip('[]').replace(" ", "").split(','))
        lst = list(map(lambda x: int(x) , lst))

    return lst
    
def process_input(argv) :
    param_dict = {}
    
    if (argv[1] == "help" or argv[1] == "-h"):
        usage()
        sys.exit(0)
        
    elif (len(sys.argv) != 8):
        print("All input required. If you want it as basic, put '-n'. Current input number: {input}".format(input = len(sys.argv)))
        usage()    
    
    # Transform String list argument into readable int list
    for i in range(1, len(argv)):
        param_dict[i] = parse_input(i, argv)
    print(param_dict)
    
    
    img, observation, mode, alpha, num_cell, cell_sz, sparse_freq = param_dict.values()

    
    #Make sure the data type input is in correct format
    assert type(img) == str
    assert (type(observation) == str or type(observation) == list)
    assert type(mode) == str
    assert (type(alpha) == list or type(alpha) == int or type(alpha) == float)
    assert (type(num_cell) == list or type(num_cell) == int or type(num_cell) == float)
    if (observation == 'V1'):
        assert (type(cell_sz) == list or type(cell_sz) == int)
        assert (type(sparse_freq) == list or type(sparse_freq) == float)
    
    return img, observation, mode, alpha, num_cell, cell_sz, sparse_freq
    
def run_sim(method, observation, mode, rep, alpha, num_cell, img_arr):
    dim = img_arr.shape
    if (len(dim) == 3) :
    	n, m, rgb = dim
    else :
    	n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    img_arr = np.array([img_arr]).squeeze()
    print("Variable Received:\nobservation={obs}\nmode={mode}\nalpha={alp}\nnum_cell={num}\n".format(obs = observation, mode = mode, alp = alpha, num = num_cell))
    print(mode)
    
    reconst = filter_reconstruct(img_arr, num_cell = num_cell, alpha = alpha, method = method, observation = observation, mode = mode)
    
    # Call function and calculate error
    error = error_calculation(img_arr, reconst)
    return error

def run_sim_V1(method, observation, mode, rep, alpha, num_cell, cell_size, sparse_freq, img_arr):
    dim = img_arr.shape
    if (len(dim) == 3) :
    	n, m, rgb = dim
    else :
    	n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    img_arr = np.array([img_arr]).squeeze()
    reconst = filter_reconstruct(img_arr, num_cell = num_cell, cell_size=cell_size, sparse_freq=sparse_freq, alpha = alpha, method = method, observation = observation, mode = mode)
    error = error_calculation(img_arr, reconst)
    print("Variable Received:\nrep={rep}\nobservation={obs}\nmode={mode}\nalpha={alp}\nnum_cell={num}\ncell_sz={sz}\nsparse_freq={freq}\n\nERROR={error}\n".format(rep = rep, obs = observation, mode = mode, alp = alpha, num = num_cell, sz=cell_size, freq=sparse_freq, error = error))
    return error

def main() :
    '''
    arg[1] = img_name ex)tree_part1.jpg
    arg[2] = observation_type ex)pixel, V1, gaussian
    arg[3] = mode ex)color/black
    arg[4] = alpha_list ex) [0.001, 0.01, 0.1]
    arg[5] = num_cell list ex) [50, 100, 200, 500]
    arg[6] = cell_size list ex) [1, 2, 4, 8, 10]
    arg[7] = sparse_freq ex) [1, 2, 4, 8, 10]
    '''
    # Set up hyperparameters that would affect results
    param_dict = {}
    
    img, observation, mode, alpha_list, num_cell, cell_size, sparse_freq = process_input(sys.argv)
    delay_list = []
    params = []
    method = 'dct'
    rep = np.arange(20)


    image_nm = img.split('.')[0]
    img_arr = process_image(img, mode)
    
    # Call dask
    client = Client()
    if (observation.upper() != 'V1') :
        search_list = [rep, alpha_list, num_cell]

        # All combinations of hyperparameter to try
        search = list(itertools.product(*search_list))
        search_df = pd.DataFrame(search, columns= [ 'rep', 'alp', 'num_cell'])
        print(search_df.head())

        sim_wrapper = lambda rep, alp, num_cell: run_sim(method, observation, mode, rep, alp, num_cell, img_arr)

        for p in search_df.values:
            delay = dask.delayed(sim_wrapper)(*p)
            delay_list.append(delay)

    elif (observation.upper() == 'V1'):
        search_list = [rep, alpha_list, num_cell, cell_size, sparse_freq]

        # All combinations of hyperparameter to try
        search = list(itertools.product(*search_list))
        search_df = pd.DataFrame(search, columns= ['rep', 'alp', 'num_cell', 'cell_size', 'sparse_freq'])
        print(search_df.head())
        sim_wrapper = lambda rep, alp, num_cell, cell_size, sparse_freq: run_sim_V1(method, observation, mode, rep, alp, num_cell, cell_size, sparse_freq, img_arr)

        for p in search_df.values:
            delay = dask.delayed(sim_wrapper)(*p)
            delay_list.append(delay)

    else :
        print("The observation {observation} is currently not supported. Please try valid observation type.".format(
            observation = observation))

    print('running dask completed')

    print('running dask completed')

    futures = dask.persist(*delay_list)
    print('futures completed')
    progress(futures)
    print('progressing futures')

    # Compute the result
    results = dask.compute(*futures)
    print('result computed')
    results_df = pd.DataFrame(results, columns=['error'])#, 'theta', 'reform', 's'])
    param_csv_nm = "param_" + "_".join(str.split(time.ctime().replace(":", "_")))
    param_path = data_save_path(image_nm, method, observation, '{mode}_{param_csv_nm}'.format(mode = mode, param_csv_nm = param_csv_nm))

    # Add error onto parameter
    params_result_df = search_df.join(results_df['error'])

    # save parameter_error data with error_results data
    params_result_df.to_csv(param_path)

    print("Execution Completed and file saved")

    hyperparam_track = data_save_path(image_nm, method, observation, '{mode}_hyperparam'.format(mode = mode))
    hyperparam_track = hyperparam_track.split('.')[0] + '.txt'
    f = open(hyperparam_track, 'a+')
    hyperparam_list = list(zip(search_df.columns, search_list))
    f.write(f"{param_csv_nm}\n")
    for hyperparam in hyperparam_list :
        f.write(f"   {hyperparam[0]}: {hyperparam[1]}\n")
    f.write("\n\n")
    f.close()
    print("Saved hyperparameter to the txt")


if __name__ == "__main__":
    main()
