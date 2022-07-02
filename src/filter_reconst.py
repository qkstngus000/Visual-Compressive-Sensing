import numpy as np
from src.structured_random_features.src.models.weights import V1_weights
from src.V1_Compress import generate_Y, compress
import os.path


def filter_reconstruction(num_cell, img_arr, cell_size, sparse_freq, filter_dim = (30, 30), alpha = None, rand_weight = False) :
    
    #alpha parameter is dependent on the number of cell if alpha is not specified
    if (alpha == None) :
        alpha = 1 * 50 / num_cell
    
    # Retrieve image dimension
    n, m = img_arr.shape
    
    # Create Filter
    filt = np.zeros(filter_dim)
    filt_n, filt_m = filter_dim
        
    # Fix the V1 weights if the random_weight parameter is set to be true
    if (rand_weight == True):
        W = V1_weights(num_cell, filter_dim, cell_size, sparse_freq) 

    # Preprocess image and add zeros so the cols and rows would fit to the filter for any size
    new_n = n + (filt_n - (n % filt_n))
    new_m = m + (filt_m - (m % filt_m))

    img_arr_aug = np.zeros((new_n, new_m))
    img_arr_aug[:n, :m] = img_arr

    print(img_arr_aug.shape)
    i = 1 # counter
    result = np.zeros(img_arr.shape)
    cur_n, cur_m = (0, 0)
    num_work = (new_n * new_m) // (filt_n * filt_m)
    for pt in range(num_work):
        if (i % (num_work // 5) == 0) :
            print("iteration", i)
        # Randomize V1 weights for each batch if random_weight param is set to false
        if (rand_weight != True) :
            W = V1_weights(num_cell, filter_dim, cell_size, sparse_freq) 

        # keep track over height of the batches
        if (cur_m >= new_m) :
            cur_n += filt_n
            cur_m = 0

        nxt_m = cur_m + filt_m
        pt = img_arr_aug[cur_n : (cur_n + filt_n), cur_m : nxt_m]

        y = generate_Y(W, pt)
        W_model = W.reshape(num_cell, filt_n, filt_m)
        theta, reform, s = compress(W_model, y, alpha)

        img_arr_aug[cur_n : (cur_n + filt_n), cur_m : nxt_m] = reform
        cur_m = nxt_m

        i+=1

    result = img_arr_aug[:n,:m]
    return result