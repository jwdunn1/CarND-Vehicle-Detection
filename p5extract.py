#_______________________________________________________________________________
# p5extract.py                                                             80->|
# Engineer: James W. Dunn
# This module implements feature extraction routines

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


#_______________________________________________________________________________
# Customized implementation of HOG, see OpenCV and scikit-image versions
def histoGrad(img, binCount=12, cellSize=4):
    binCells,magCells= [],[]
    cx= cy= cellSize
    gx,gy= cv2.Sobel(img,cv2.CV_32F,1,0), cv2.Sobel(img,cv2.CV_32F,0,1) # gradients in x and y direction
    magnitude,angle= cv2.cartToPolar(gx,gy)
    bins= np.int32(binCount*angle*(1-1e-7)/(2*np.pi)) #scale back angle to avoid overbinning
    for i in range(0,int(img.shape[0]/cy)): # group by cell
        for j in range(0,int(img.shape[1]/cx)):
            binCells.append(bins[i*cy:i*cy+cy, j*cx:j*cx+cx])
            magCells.append(magnitude[i*cy:i*cy+cy, j*cx:j*cx+cx])
    hist= np.hstack([np.bincount(i.ravel(), j.ravel(), binCount) for i,j in zip(binCells, magCells)])
    return np.sqrt(hist/(hist.sum()+1e-8)) #  L1-sqrt method


#_______________________________________________________________________________
# Function to return HOG features
def get_hog_features(img, orient, pix_per_cell):
    features= histoGrad( img, orient, pix_per_cell )
    return features.reshape((orient*pix_per_cell*pix_per_cell,))    #192

# Function to return color histogram features  
def get_color_hist(img, nbins=32, bins_range=(0, 32)):
    return np.histogram(img, bins=nbins, range=bins_range)

# Function to return spatial binning of color features
def get_spatial(img, size=(4,4)):
    return np.ravel(cv2.resize(img, size, interpolation=3)) #CV_INTER_AREA

# Function to return texture features
def get_texture_features(img, quantization=8, radius=3):
    lbp= np.ravel(local_binary_pattern(img, quantization, radius, method='uniform'))
    return np.mean(lbp.reshape(-1, 4), axis=1) # average pooling



