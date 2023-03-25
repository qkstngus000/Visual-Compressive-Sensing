import sys
import numpy as np

def usage():
    print(
        '''Usage
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
    )
    sys.exit(0)

def parse_input(i, argv):
    default = {
        2 : 'dct',
        3 : 'db2',
        4 : 4,
        5 : 'classical',
        6 : [0.001, 0.01, 0.1],
        7 : [50, 100, 200, 500],
        8 : np.arange(1, 10, 1),
        9 : np.arange(1, 10, 1)
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
        
    elif (i == 5):
        return argv[i]
    
    elif (i == 6):
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
    

def main():
    #variables needed
    #print(len(sys.argv))
    #lst = None
    if (True):
        lst = 1
    print(lst)
#     process_input(sys.argv)
    
    return 0;
if __name__ == "__main__":
    main()
