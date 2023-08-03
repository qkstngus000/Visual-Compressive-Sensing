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
        arg[3] = mode ex)color/black
        arg[4] = dwt_type ex) haar, db1, db2, etc
        arg[5] = lv ex) [1, 2, 3, 4, 5]
        arg[6] = alpha_list ex) [0.001, 0.01, 0.1]
        arg[7] = num_cell list ex) [50, 100, 200, 500] or [0.1, 0.3, 0.5, 0.7]
        arg[8] = cell_size list ex) [1, 2, 4, 8, 10]
        arg[9] = sparse_freq ex) [1, 2, 4, 8, 10]
        '''
    )
    sys.exit(0)

def parse_input(i, argv):
    default = {
        1 : 'tree_part1.jpg',
        2 : 'pixel',
        3 : 'black',
        4 : 'db2',
        5 : [1, 2, 3, 4, 5],
        6 : list(np.logspace(-3, 3, 7)),
        7 : [50, 100, 200, 500],
        8 : list(np.arange(1, 11, 1)),
        9 : list(np.arange(1, 11, 1))
        }
    
    ## Checking statement if default is formed correctly
    #print("In parse_input, default {i} is {default}".format(i = i, default = default.get(i)))
    
    if (i != 1 and (argv[i] == 'n' or argv[i] == '-n')):
        return default.get(i)
    
    if (i == 1):
        assert argv[i] != 'n'
        return argv[i]
    elif (i == 2):
        lst = list(argv[i].strip('[]').replace(" ", "").split(','))
        return lst
    elif (i == 3) :
        if argv[i].lower() in ['-c', 'color']:
            return "color"
        return default.get(i)
    elif (i == 4):
        return argv[i]
    elif (i == 6) :
        lst = list(argv[i].strip('[]').replace(" ", "").split(','))
        print(lst)
        lst = list(map(lambda x: float(x) , lst))
    elif (i == 7) :
        lst = lst = list(argv[i].strip('[]').replace(" ", "").split(','))
        if (lst[0].isdigit):
            lst = list(map(lambda x: int(x) , lst))
        else:
            lst = list(map(lambda x: float(x) , lst))
    else:
        lst = list(argv[i].strip('[]').replace(" ", "").split(','))
        lst = list(map(lambda x: int(x) , lst))
        
    if ((type(lst) == list or type(lst).__module__ == np.__name__) and len(lst) == 1):
        return lst[0]
    else:
        return lst
    
def process_input(argv) :
    param_dict = {}
    
    if (argv[1] == "help" or argv[1] == "-h"):
        usage()
        sys.exit(0)
        
    elif (len(sys.argv) != 10):
        print("All input required. If you want it as basic, put '-n'. Current input number: {input}".format(input = len(sys.argv)))
        usage()    
    
    # Transform String list argument into readable int list
    for i in range(1, len(argv)):
        param_dict[i] = parse_input(i, argv)
    print(param_dict)
    
    
    img, observation_lst, mode, dwt_type, lv, alpha, num_cell, cell_sz, sparse_freq = param_dict.values()
    print(dwt_type)
    
    #Make sure the data type input is in correct format
    assert type(img) == str
    assert (type(observation_lst) == str or type(observation_lst) == list)
    assert type(dwt_type) == str
    assert type(mode) == str
    assert type(lv) == list or type(lv) == int
    assert (type(alpha) == list or type(alpha) == int or type(alpha) == float)
    assert (type(num_cell) == list or type(num_cell) == int or type(num_cell) == float)
    assert (type(cell_sz) == list or type(cell_sz) == int)
    assert (type(sparse_freq) == list or type(sparse_freq) == float)
    
    return img, observation_lst, mode, dwt_type, lv, alpha, num_cell, cell_sz, sparse_freq
    
def run_sim(method, observation, mode, dwt_type, rep, lv, alpha, num_cell, img_arr):
    dim = img_arr.shape
    if (len(dim) == 3) :
    	n, m, rgb = dim
    else :
    	n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    rep = int(rep)
    lv = int(lv)
    alpha = float(alpha)
    
    img_arr = np.array([img_arr]).squeeze()
    
    reconst = filter_reconstruct(img_arr, num_cell = num_cell, alpha = alpha, method = method, observation = observation, mode = mode, lv = lv, dwt_type = dwt_type)
    
    # Call function and calculate error
    error = error_calculation(img_arr, reconst)
    
#     return error, theta, reconst, s
    return error

def run_sim_V1(method, observation, mode, dwt_type, rep, lv, alpha, num_cell, cell_size, sparse_freq, img_arr):
    dim = img_arr.shape
    if (len(dim) == 3) :
    	n, m, rgb = dim
    else :
    	n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    lv = int(lv)
    rep = int(rep)
    alpha = float(alpha)
    
    img_arr = np.array([img_arr]).squeeze()
    print("Variable Received:\nobservation={obs}\nmode={mode}\nalpha={alp}\nnum_cell={num}\ncell_sz={sz}\nsparse_freq={freq}".format(obs = observation, mode = mode, alp = alpha, num = num_cell, sz=cell_size, freq=sparse_freq))
    
    #Filter reconst to make sure it can reconstruct any size 
    reconst = filter_reconstruct(img_arr, num_cell = num_cell, cell_size=cell_size, sparse_freq=sparse_freq, alpha = alpha, method = method, observation = observation, mode = mode, lv = lv, dwt_type = dwt_type)
    
    # Calculates for the error per pixel
    error = error_calculation(img_arr, reconst)
    
#     return error, theta, reconst, s
    return error

def main() :
    '''
    arg[1] = img_name ex)tree_part1.jpg
    arg[2] = observation_type ex)pixel, V1, gaussian
    arg[3] = mode ex)color/black
    arg[4] = dwt_type ex) n (none for dct), haar, db1, db2, etc
    arg[5] = lv ex) n (none for dct), [1, 2, 3, 4, 5]
    arg[6] = alpha_list ex) [0.001, 0.01, 0.1]
    arg[7] = num_cell list ex) [50, 100, 200, 500]
    arg[8] = cell_size list ex) [1, 2, 4, 8, 10]
    arg[9] = sparse_freq ex) [1, 2, 4, 8, 10]
    '''
    # Parse given hyperparameters that would affect results
    img, observation_lst, mode, dwt_type, lv, alpha_list, num_cell, cell_size, sparse_freq = process_input(sys.argv)
    delay_list = []
    params = []
    method = 'dwt'
    rep = np.arange(20)


    image_nm = img.split('.')[0]
    img_arr = process_image(img, mode)
    
    for observation in observation_lst:
        # Call dask
        client = Client()

        if (observation.upper() != 'V1') :
            search_list = [rep, lv, alpha_list, num_cell]

            # All combinations of hyperparameter to try 
            search = list(itertools.product(*search_list))             
            search_df = pd.DataFrame(search, columns= [ 'rep', 'lv', 'alp', 'num_cell'])
            print(search_df.head())

            # counter = 0; # Keep track of number of iteration. Debugging method
            for p in search_df.values:
                delay = dask.delayed(run_sim)(method, observation, mode, dwt_type, *p, img_arr)
                delay_list.append(delay)

        elif (observation.upper() == 'V1'):
            search_list = [rep, lv, alpha_list, num_cell, cell_size, sparse_freq]

            # All combinations of hyperparameter to try 
            search = list(itertools.product(*search_list))             
            search_df = pd.DataFrame(search, columns= [ 'rep', 'lv', 'alp', 'num_cell', 'cell_size', 'sparse_freq'])
            print(search_df.head())
            for p in search_df.values:
                delay = dask.delayed(run_sim_V1)(method, observation, mode, dwt_type, *p, img_arr)
                delay_list.append(delay)

        else :
            print("The observation {observation} is currently not supported. Please try valid observation type.".format(
                observation = observation))

        print('running dask completed')

        futures = dask.persist(*delay_list)
        print('futures completed')
        progress(futures)
        print('progressing futures')

        # Compute the result
        results = dask.compute(*futures)
        print('result computed')
        results_df = pd.DataFrame(results, columns=['error'])
        param_csv_nm = "param_{dwt_type}_".format(dwt_type = dwt_type) + "_".join(str.split(time.ctime().replace(":", "_")))
        param_path = data_save_path(image_nm, method, observation, param_csv_nm)
    #     result_path = data_save_path(image_nm, method, observation, "result_" + "_".join(str.split(time.ctime().replace(":", "_"))))
        # Add error onto parameter
        params_result_df = search_df.join(results_df['error'])

        # save parameter_error data with error_results data
        params_result_df.to_csv(param_path)
    #     results_df.to_csv(result_path)
        print("Execution Completed and file saved")

        hyperparam_track = data_save_path(image_nm, method, observation, 'hyperparam')
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
