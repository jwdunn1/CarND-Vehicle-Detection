#_______________________________________________________________________________
# p5pipeline.py                                                            80->|
# Engineer: James W. Dunn
# This module implements the video pipeline

import cv2
import glob
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.externals import joblib
from keras.models import model_from_json
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import local_binary_pattern
from p5extract import get_hog_features # local feature extraction routines
from p5extract import get_spatial

#_______________________________________________________________________________
# Globals

diagScreen= None  # annotation overlay (diagram screen)
cellBatch= [[], [], [], [], []]  # cell, x, y, s, prediction
lastDiagScreen= None
box_list= []
model= None
frameCount= 0


#_______________________________________________________________________________
# Load trained binary classifier model and scaler

with open('model.json', 'r') as jfile:
    model = model_from_json(json.loads(jfile.read()))
model.compile("adam", "binary_crossentropy")
#weights_file = args.model.replace('json', 'h5')
model.load_weights('model.h5')
model.summary()
X_scaler= joblib.load('scaler.dat')


#_______________________________________________________________________________
# Assemble feature vector from HOG and spatial binning of color

def extractFeaturesFromCell(cell, orient=12, pix_per_cell=4):
    featureVector = []
    img= cv2.cvtColor(cell, cv2.COLOR_RGB2YCrCb) 
    for channel in range(img.shape[2]):
        # Compute HOG and coarse spatial binning of color
        featureVector.append(np.concatenate((
            get_hog_features(img[:,:,channel], orient, pix_per_cell),
            get_spatial(img[:,:,channel], size=(4,4))/255.
            ))) # Numerically scale to range (0,1)
    return np.ravel(featureVector)


#_______________________________________________________________________________
# Numerical scaling of cell

def scaleCell(cell):
    global X_scaler # use previously defined scaler
    X= cell.astype(np.float64).reshape((1, -1))                       
    # Apply the scaler to X
    return np.array(X_scaler.transform(X))


#_______________________________________________________________________________
# Sliding windows definition

x= np.linspace(0, 14, 15)
y=np.array([766,780,796,814,834,857,882,910,941,976,1014,1057,1103,1155,1214])
coefs1= np.polyfit(x,y,3)
y=np.array([411,410,410,409,409,408,407,406,406,405,403,402,401,398,397])
coefs2= np.polyfit(x,y,2)
y=np.array([14,16,18,20,23,25,28,31,35,38,43,46,52,59,66])
coefs3= np.polyfit(x,y,2)
def transformIdxToScreen(idxFloat):
    global coefs1, coefs2, coefs3
    f1= np.poly1d(coefs1)
    f2= np.poly1d(coefs2)
    f3= np.poly1d(coefs3)
    return f1(idxFloat), f2(idxFloat), f3(idxFloat)


#_______________________________________________________________________________
# Bounding box renderer (modified code from lecture notes)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected boxes
    for box_number in range(1, labels[1]+1):
        # Find pixels with each box_number label value
        nonzero= (labels[0] == box_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy= np.array(nonzero[0])
        nonzerox= np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox= ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        # Draw the box on the image
        # RULE: if bbox is over 2.8:1 aspect ratio, split in two
        bWidth= bbox[1][0]-bbox[0][0]    #x1-x0
        bHeight= bbox[1][1]-bbox[0][1]   #y1-y0
        if bHeight>100 and bWidth/bHeight>2.68:
            mid= ((bbox[0][0]+bWidth//2+5,bbox[0][1]), (bbox[0][0]+bWidth//2-5,bbox[1][1]))
            cv2.rectangle(img, bbox[0], mid[1], (0,150,255), 5)
            cv2.rectangle(img, mid[0], bbox[1], (0,150,255), 5)
        else:
            # RULE: do not render boxes of height less than min cell size
            if bHeight>=14 and bWidth>=14:
                cv2.rectangle(img, bbox[0], bbox[1], (0,150,255), 5)
    return img

def addBoxToList(x,y,s): #add a screen box for each of the positive cells
    global box_list
    box_list.append( ((x, y), (x+s+5, y+s+5)) ) # make the cells slightly larger so they overlap

def drawData(img):
    global box_list
    heat= np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat= add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat= apply_threshold(heat,1)

    # Find final boxes from heatmap using label function
    labels= label(heat)
    draw_labeled_bboxes(img, labels)
    
    # Trim the box list
    boxQLEN= 50
    if len(box_list)>boxQLEN: # keep the last n boxes
        top= len(box_list)-boxQLEN
        for i in range(0,top): # circular queue 
            del box_list[0]


#_______________________________________________________________________________
# Cell support routines for resizing, feature extraction, and batching

def resizeCell(img):
    if len(img[0]) != 16:
        if len(img[0])>16:
            cell= cv2.resize(img, (16,16), interpolation=3) #CV_INTER_AREA
        else:
            cell= cv2.resize(img, (16,16), interpolation=2) #CV_INTER_CUBIC
    else:
        cell= img
    return cell

def prepareCell(c):
    cell= resizeCell(c)
    fCell= extractFeaturesFromCell(cell)
    sCell= scaleCell(fCell)
    return sCell

def addCell(c,x,y,s):
    global cellBatch
    cellBatch[0].append(c)
    cellBatch[1].append(x)
    cellBatch[2].append(y)
    cellBatch[3].append(s)


#_______________________________________________________________________________
# Process each vertical strip of cells in the NLS matrix

def processStrip(img, i):
    x,y,s= transformIdxToScreen(i)
    for i in range(4):
        addCell(img[y:y+s, x:x+s, :],x,y,s)
        y= y+s


#_______________________________________________________________________________
# Prepare and send a batch of cells to the GPU for predictions, then postprocess

def processBatch(baseImage): 
    global cellBatch, diagScreen, lastDiagScreen
    prepBatch= []
    for i in range( len(cellBatch[0]) ):
        prepBatch.append( prepareCell(cellBatch[0][i]) )
    predictionResults= model.predict( np.array(prepBatch).reshape((60,-1)) )
    cellBatch[4]= np.array(predictionResults>0.9,dtype=np.uint8)
    for i in range( len(cellBatch[4]) ):
        x= cellBatch[1][i]
        y= cellBatch[2][i]
        s= cellBatch[3][i]
        if cellBatch[4][i]:
            addBoxToList(x,y,s)
    drawData(diagScreen)
    lastDiagScreen= np.copy(diagScreen)


#_______________________________________________________________________________
# Alternate frame redraw

def redrawBatch(baseImage):
    global cellBatch, diagScreen, lastDiagScreen
    diagScreen= np.copy(lastDiagScreen)
    cellBatch= [[], [], [], [], []]  # reset batch: cell, x, y, s, prediction

#_______________________________________________________________________________
# Main process for each frame

def process(img):
    global diagScreen, frameCount
    diagScreen= np.zeros_like(img).astype(np.uint8)
    if frameCount%2==0:
        for i in range(0,15):
            processStrip(img, i) # add cells to batch
        processBatch(img)
    else:
        redrawBatch(img)

    cv2.putText(img, str(frameCount), (1200,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    result= cv2.addWeighted(img, 1, diagScreen, 1, 0) # annotate the original
    frameCount+= 1
    return result

#_______________________________________________________________________________
# Executive

def procVideo(fileName):
    global frameCount
    clip= VideoFileClip(fileName)
    imgName= fileName.split('/')[1]
    project_video_output= 'output_images/'+imgName
    print('Processing video...')
    project_video_clip= clip.fl_image(process)
    t=time.time()
    project_video_clip.write_videofile(project_video_output, audio=False)
    t2 = time.time()
    print(round(t2-t, 2), 'seconds to complete @', round(frameCount/(t2-t), 2), 'fps' )

procVideo('video/project_video.mp4')  # <---Insert video to process here