# Signal Reconstruction Research
Description: Compressed sensing (CS) is a widely accepted theory for signal-based data recovery using trained samples, which enables the reconstruction of a full signal at a low cost. Typically, pixel-based sampling techniques are utilized, but in this study, we propose a new observation technique using a biological model, the Visual Cortex (V1), to acquire more robust and improved signal reconstruction. The V1 is a biological model that has been shown to be effective in processing visual information in the brain. We hypothesize that this biological approach to observation combined with LASSO sparse coding prediction could lead to a better minimization of reconstruction errors compared to the traditional mathematical approach. Thus, we compare the effectiveness of the signal reconstruction between the V1 model and two classical models, namely pixel selection and Gaussian, to determine which method performs best. This study aims to highlight the potential of utilizing biological models in CS and to provide a better understanding of the performance of different observation techniques.

## Table of Contents
1. [Installation](#installation)
2. [Main Functions](#main-functions)
3.
4.
5.

## Installation
Assuming you have python environment, fork this project into desired directory.

## Main Functions

### src
This is where all source codes are stored. Source code mainly separated into three different file. 
    
&ensp; * Compress Sensing Library: Contains all codes that deals with computations of signal data with its observation method. 
&ensp; * (hyperparam sweep file): Calls Signal reconstruction method defined in Compress Sensing Library and call dask (parallel computation library) to compute hyperparameters that user wants to test. There are total two hyperparameter sweep files, which one deals with descrete cosine transform ([DCT](https://en.wikipedia.org/wiki/Dual-clutch_transmission)) and the other deals with descrete wavelet transform ([DWT](https://en.wikipedia.org/wiki/Discrete_wavelet_transform)). 
&ensp; * 

### result
This is where all hyperparameter sweeped data is getting stored. Once hyperparameter sweep function is run, the source code will find the result directory with its save path and save it into correspondent path.

### figures
This is where figures generated from result data is getting stored

### structured_random_features
This is the base of our project where folder contains neural-network V1 model the generate V1 observation

### 