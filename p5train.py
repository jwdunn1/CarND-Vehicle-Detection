#_______________________________________________________________________________
# p5train.py                                                               80->|
# Engineer: James W. Dunn
# This module trains using files in the 'training' folder

import os
import cv2
import json
import glob
import time
import errno
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from p5extract import get_hog_features # local feature extraction routines
from p5extract import get_spatial


#_______________________________________________________________________________
# Metaparameters
orient= 12
pix_per_cell= 4


#_______________________________________________________________________________
# Function to extract features from a list of images and/or cells

def extract_features(imgs, cells=True, orient=12, pix_per_cell=4):
    # Create a list to append feature vectors
    features= []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image= mpimg.imread(file)  #inbound format is range 0-1
        # apply color conversion
        feature_image= cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        if cells: # Already sized at 16x16 cells
            hog_features= []
            for channel in range(feature_image.shape[2]):
                # Train on gradient AND resized color patch
                hog_features.append(np.concatenate((get_hog_features(feature_image[:,:,channel], orient, pix_per_cell), 
                    np.ravel(cv2.resize(feature_image[:,:,channel], (4,4), interpolation=3) ) ))) #CV_INTER_AREA
                
            hog_features= np.ravel(hog_features)        
            # Append the new feature vector to the features list
            features.append(hog_features)
        else:
            #first, slice the 64x64 image into quads by column
            quad= []
            quad.append(feature_image[:, 0:16,:])
            quad.append(feature_image[:,16:32,:])
            quad.append(feature_image[:,32:48,:])
            quad.append(feature_image[:,48:64,:])
            for i in range(4):
                #next, slice the quad into cells(16x16) by row
                square= []
                square.append(quad[i][ 0:16,:,:])
                square.append(quad[i][16:32,:,:])
                square.append(quad[i][32:48,:,:])
                square.append(quad[i][48:64,:,:])
                for j in range(4):
                    feature_image= square[j]
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        # Compute HOG and coarse spatial binning of color
                        hog_features.append(np.concatenate((
                            get_hog_features(feature_image[:,:,channel], orient, pix_per_cell), 
                            get_spatial(feature_image[:,:,channel], (4,4)) )))
                    hog_features = np.ravel(hog_features)        
                    # Append the new feature vector to the features list
                    features.append(hog_features)
            del quad, square
    # Return list of feature vectors
    return features

# load filenames of cells containing vehicles
cars = []
vimages = glob.glob('training/vehicles/JWD_vcells/*.png')
for image in vimages:
    cars.append(image)
vimages = glob.glob('training/vehicles/JWD_vcells_ch/*.png')
for image in vimages:
    cars.append(image)

# load filenames of images containing non-vehicles
notcars = []
nimages = glob.glob('training/non-vehicles/Extras/*.png')
for image in nimages:
    notcars.append(image)

del vimages, nimages


#_______________________________________________________________________________
# Load data from files into training arrays

t=time.time()
print('Extracting cells of cars...')
car_features= extract_features(cars, cells=True, orient=orient, 
                        pix_per_cell=pix_per_cell)
print('Extracting notcars images...')
notcar_features= extract_features(notcars, cells=False, orient=orient, 
                        pix_per_cell=pix_per_cell)

# load additional cells containing non-vehicles
notcars= []
nimages= glob.glob('training/non-vehicles/JWD_ncells/*.png')
for image in nimages:
    notcars.append(image)
nimages= glob.glob('training/non-vehicles/JWD_ncells_ch/*.png')
for image in nimages:
    notcars.append(image)
print('Extracting cells of notcars...')
notcarcell_features= extract_features(notcars, cells=True, orient=orient, 
                        pix_per_cell=pix_per_cell)
del nimages

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

print(np.array(car_features).shape, np.array(notcar_features).shape, np.array(notcarcell_features).shape)
# Create an array stack of feature vectors
X= np.vstack((car_features, notcar_features, notcarcell_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler= StandardScaler().fit(X)
# Apply the scaler to X
scaled_X= X_scaler.transform(X)

# Define the labels vector
y= np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)), np.zeros(len(notcarcell_features))))

# Split up data into randomized training and test sets
rand_state= np.random.randint(0, 100)
X_train, X_test, y_train, y_test= train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state)

del scaled_X, y, cars, notcars
del car_features, notcar_features, notcarcell_features
print('Using:',orient,'orientations',pix_per_cell,'pixels per cell')
print('Feature vector length:', len(X_train[0]))
print('Reserving 10% for accuracy measurement.')


#_______________________________________________________________________________
# Define a non-linear binary classification model using Keras on TensorFlow

model= Sequential([ # multi-layer non-linear with drop-out regularization
    Dropout(.5, input_shape=(len(X_train[0]),) ),
    Dense(255, init='normal', activation='relu'),
    Dropout(.5),
    Dense(127, init='normal', activation='relu'),
    Dropout(.5),
    Dense(1, init='normal', activation='sigmoid')
])

# binary classification
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


#_______________________________________________________________________________
# Training loop

# Check the training time for the GPU-based model
print('Now training for 35 epochs...')
t=time.time()
model.fit(X_train, y_train, nb_epoch=35, batch_size=32, shuffle=True, verbose=2)
t2 = time.time()
print(round(t2-t, 2), 'seconds to train the model...')

print('Test accuracy of model = ', round(model.evaluate(X_test, y_test, batch_size=16)[1], 4), ' <--------------------')

# Finish the training cycle using the test data (to learn even more!)
print('Finish the training cycle using the test data for 21 epochs...')
t=time.time()
model.fit(X_test, y_test, nb_epoch=21, batch_size=32, shuffle=True, verbose=2)
t2 = time.time()
print(round(t2-t, 2), 'seconds to complete the model...')

# Clean up memory
del X_train, X_test, y_train, y_test

#_______________________________________________________________________________
# Save the model, weights, and scaler for later use

def deleteFile(file):
    try:
        os.remove(file)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise

deleteFile('model.json')
deleteFile('model.h5')
jsonString= model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(jsonString, outfile)
model.save_weights('model.h5')
joblib.dump(X_scaler, 'scaler.dat')
del model, jsonString, X_scaler
print('Model saved to model.json, weights saved to model.h5, scaler saved to scaler.dat')
