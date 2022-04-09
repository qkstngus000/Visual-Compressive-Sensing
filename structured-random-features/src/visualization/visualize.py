import sys
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import seaborn as sns

sys.path.append('../models')
from networks import alexnet

model_fn = '../models/checkpoints_structured/model_best.pth.tar'

def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_k = k[7:]
            new_state_dict[new_k] = v

    return new_state_dict


#lots of plotting code borrowed from https://github.com/Niranjankumar-c/DeepLearning-PadhAI/blob/master/DeepLearning_Materials/6_VisualizationCNN_Pytorch/CNNVisualisation.ipynb
def plot_filters_single_channel_big(t):
    #setting the rows and columns
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]

    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)
    npimg = npimg.T
    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))    
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False,
                          cmap='gray', ax=ax, cbar=False)


def plot_filters_multi_channel(t):
    #get the number of kernals
    num_kernels = t.shape[0]    
    
    #define number of columns for subplots
    num_cols = 8
    #rows = num of kernels
    num_rows = num_kernels // num_cols
    
    #set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))
    
    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        
        #for each kernel, we convert the tensor to numpy 
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
    plt.savefig('myimage.png', dpi=100)    
    plt.tight_layout()
    plt.show()


def plot_filters_single_channel(t):
    #kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12
    
    nrows = 1 + nplots//ncols
    #convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)
    
    count = 0
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols, nrows))
    
    #looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            #ax1 = fig.add_subplot(nrows, ncols, count)
            ax1 = axs[count]
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            count += 1
            
    plt.tight_layout()
    plt.show()    

def plot_weights(model, layer_num, single_channel = True, collated = False):
  #extracting the model features at the particular layer number
  layer = model.features[layer_num]
  #checking whether the layer is convolution layer or not 
  if isinstance(layer, nn.Conv2d):
      #getting the weight tensor data
    weight_tensor = model.features[layer_num].weight.data
    if single_channel:
      if collated:
          plot_filters_single_channel_big(weight_tensor)
      else:
          plot_filters_single_channel(weight_tensor)
    else:
        if weight_tensor.shape[1] == 3:
            plot_filters_multi_channel(weight_tensor)
        else:
            print("Can only plot weights with three channels")
  else:
      print("Can only visualize layers which are convolutional")


model = alexnet()
checkpoint = torch.load(model_fn)
checkpoint['state_dict'] = fix_state_dict(checkpoint['state_dict'])
model.load_state_dict(checkpoint['state_dict'])
plot_weights(model, 3, single_channel = True, collated=False)
