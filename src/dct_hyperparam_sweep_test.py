import numpy as np
import numpy.linalg as la
import sys
sys.path.append("../")
import matplotlib.pyplot as plt

# Package for importing image representation
from PIL import Image, ImageOps

from src.compress_sensing_library import *
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
        arg[2] = observation_type ex)classical, V1, gaussian
        arg[3] = alpha_list ex) [0.001, 0.01, 0.1]
        arg[4] = num_cell list ex) [50, 100, 200, 500]
        arg[5] = cell_size list ex) [1, 2, 4, 8, 10]
        arg[6] = sparse_freq ex) [1, 2, 4, 8, 10]
        '''
    )
    sys.exit(0)

def parse_input(i, argv):
    default = {
        1 : 'tree_part1.jpg'
        2 : 'classical',
        3 : [0.001, 0.01, 0.1],
        4 : [50, 100, 200, 500],
        5 : np.arange(1, 10, 1),
        6 : np.arange(1, 10, 1)
        }
    if (i != 1 and argv[i] == 'n'):
        return default[i]
    
    if (i == 1):
        assert argv[i] != 'n'
        return argv[i]
    elif (i == 2):
        return argv[i]
    elif (i == 3) :
        lst = list(argv[i].strip('[]').replace(" ", "").split(','))
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
        print("All input required. If you want it as basic, put 'n'")
        usage()    
    
    # Transform String list argument into readable int list
    for i in range(1, len(argv)):
        param_dict[i] = parse_input(i, argv)
    print(param_dict)
    file, method, observation, alpha, num_cell, cell_sz, sparse_freq = param_dict.values()
    
    #Make sure the data type input is in correct format
#     assert type(file) == str
#     assert type(method) == str
#     assert type(observation) == str
#     #assert type(alpha) == list
#     assert type(num_cell) == list
#     assert type(cell_sz) == list
#     assert type(sparse_freq) == list
    
    return
    
def run_sim(method, observation, rep, alpha, num_cell, lv =  img_arr):
    dim = img_arr.shape
    n, m = dim
    # Deal with fraction num_cell amount
    if (num_cell < 1):
        num_cell = round(n * m * num_cell)
    num_cell = int(num_cell)
    img_arr = np.array([img_arr]).squeeze()
    
    if (observation.lower() == 'classical') :
        W, y = generate_classical_variables(img_arr, num_cell)
        
    elif (observation.lower() == 'gaussian') :
        W, y = generate_gaussian_variables(img_arr, num_cell)
    
    elif (observation.upper() == 'V1') :
        W, y = generate_V1_variables(img_arr, num_cell, cell_size, sparse_freq)
        
    else :
        print("This obervation technique is currently not supported\n Please use valid observation: ['classical', 'gaussian', 'V1']")
    
    # Call function and calculate error
    theta, reconst, s = reconstruct(W, y, alpha, method = method, lv = lv, dwt_type = dwt_type)
    error = error_calculation(img_arr, reconst)
    
    return error, theta, reconst, s

def main() :
    '''
    arg[1] = img_name ex)tree_part1.jpg
    arg[2] = method ex) dct, dwt
    arg[3] = dwt_type ex) n (none for dct), harr, db1, db2, etc
    arg[4] = lv ex) n (none for dct), [1, 2, 3, 4, 5]
    arg[5] = observation_type ex)classical, V1, gaussian
    arg[6] = alpha_list ex) [0.001, 0.01, 0.1]
    arg[7] = num_cell list ex) [50, 100, 200, 500]
    arg[8] = cell_size list ex) [1, 2, 4, 8, 10]
    arg[9] = sparse_freq ex) [1, 2, 4, 8, 10]
    '''
    # Set up hyperparameters that would affect results
    param_dict = {
        #1 : None, 
        #2 : 'dct',
        #3 : 'classical', 
        #4 : np.logspace(-3, 3, 7),
        #5 : [50, 100, 200, 500], 
        #6 : [2, 5, 7],
        #7 : [1, 2, 5]
    }
    
    
    image_path ='../image/{img}'.format(img = file)
    delay_list = []
    params = []
    alpha = np.logspace(-3, 3, 7)
    rep = np.arange(20)


    image_nm = image_path.split('/')[2].split('.')[0]
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img_arr = np.asarray(img)
    #plt.imshow(img_arr)
    #plt.show()
    save_path = os.path.join("../result/{img_nm}/V1".format(img_nm = image_nm))

    # TODO: if classical, change search_list only to use rep, alpha, num_cell
    
    if (observation.lower() == 'classical' or observation.lower() == 'gaussian') :
        search_list = [rep, alpha, num_cell]

        # All combinations of hyperparameter to try 
        search = list(itertools.product(*search_list))             
        search_df = pd.DataFrame(search, columns= [ 'rep', 'alp', 'num_cell'])
        print(search_df.head())
    
    elif (observation.upper() == 'V1'):
        search_list = [rep, alpha, num_cell, cell_sz, sparse_freq]

        # All combinations of hyperparameter to try 
        search = list(itertools.product(*search_list))             
        search_df = pd.DataFrame(search, columns= [ 'rep', 'alp', 'num_cell', 'cell_sz', 'sparse_freq'])
        print(search_df.head())
    else :
        print("The observation {observation} is currently not supported. Please try valid observation type.".format(
            observation = observation))
    # Call dask
    client = Client()

    # counter = 0; # Keep track of number of iteration. Debugging method
    for p in search_df.values:
        delay = dask.delayed(run_sim)(method, observation, *p, img_arr)
        delay_list.append(delay)

    print('running dask completed')

    futures = dask.persist(*delay_list)
    print('futures completed')
    progress(futures)
    print('progressing futures')

    # Compute the result
    results = dask.compute(*futures)
    print('result computed')
    results_df = pd.DataFrame(results, columns=['error', 'theta', 'reform', 's'])
    
    param_path = data_save_path(image_nm, method, observation, "param_" + "_".join(str.split(time.ctime().replace(":", "_"))))
    result_path = data_save_path(image_nm, method, observation, "result_" + "_".join(str.split(time.ctime().replace(":", "_"))))
    # Add error onto parameter
    params_result_df = search_df.join(results_df['error'])
    
    # save parameter_error data with error_results data
    params_result_df.to_csv(param_path)
    results_df.to_csv(result_path)
    print("Execution Complete")


if __name__ == "__main__":
    main()
