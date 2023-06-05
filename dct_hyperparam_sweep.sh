#!/bin/bash

#Directory to the python script
cd experiment

#Hyperparameter Values
img="tree_part1.jpg"
observation="classical"
alpha="-n"
num_cell='[50, 100, 200, 300, 500]'
cell_size='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]'
sparse_freq='[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]'

## For wavelet variable
lv='[1, 2, 4, 6]'
dwt_type='[harr, db1, db2]'

# Call python script
python3 dct_hyperparam_sweep_test.py $img $observation $alpha $num_cell $cell_size $sparse_freq

