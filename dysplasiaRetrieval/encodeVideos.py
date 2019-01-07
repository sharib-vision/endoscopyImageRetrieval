#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:04:00 2018

@author: shariba
"""

'''
    - extract video frames
    - list all the videos that you want to encode
    - encode using multi-threading for all the videos that has been put as array
    - only embeddings which doesnot exist are done here!!!

'''

import numpy as np
import time
import os
from imageCroppingClasses import imageDebugging

from paths import DATA_DIR, ROOT_DIR, RESULT_DIR

vFiles = ['M_11052017100716_0000000000001716_1_001_001-1_1.MP4', 'M_11012018124919_0000000000002470_1_001_001-1_1.MP4', 'M_01032018130721_0000000000003059_1_002_001-1.MP4' ]

# loop here for all vFiles
videoFileName = vFiles[2]   # vFiles[2]  is test video less images 

videoFile = os.path.join(DATA_DIR, videoFileName)
patientFileName=videoFile.split('_')[1]+'_'+videoFile.split('_')[2].strip("0")

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Video cleaning using our binary classification with DNN
    input - raw video
    output - clean video
    save as - cleanVideo/vFileName
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
outfile=videoFileName.split('.')[0]
#os.makedirs((os.path.join(ROOT_DIR, 'dysplasiaEndoscopy', 'cleanVideos')), exist_ok=True)
exists = os.path.isfile(RESULT_DIR +'/'+ outfile+'_frameScore'+'.npy')

debug = 0
runPart1 = 1

if runPart1:
    if exists:
        print('Corresponding clean videoFile already exists or its cleanFrameList exists, nothing to do!!!')
    else:
        print('Corresponding clean videoFile does not exists!!!, this will take a while')
        print('Suggestion: Try to do offline and use GPU for fast processing')
        
        from videoCleaningUsingDNN import videoCleaningWithDNN
        modelFile = os.path.join(ROOT_DIR, 'dysplasiaEndoscopy', 'binaryEndoClassifier_124_124.h5')
        frame_scores, nframes, cleanFrameList = videoCleaningWithDNN (modelFile, videoFile) 
        '''
            Identify original video frames that has been cleared for further processing
            This is 0-1 for badframe-good frame
        '''
        # save frame scores as npy array file
        # to load: np.load(outfile)
        np.save(RESULT_DIR+'/'+outfile+'_frameScore', frame_scores)
        np.save(RESULT_DIR+'/'+outfile+'_cleanFrameList', cleanFrameList)
        
        if debug:   
            im = np.reshape(cleanFrameList[0], (124,124,3))
            imageDebugging.showImageDebug(im)
            
        print('video cleaning done... saved frameListBinary and cleanFrameList as noy file, check:', RESULT_DIR)
    # write clean video into the folder with the chosen frames_scores (in binary)
    # todo save frame_scores in the cleanvideo folder for later usage 
    
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Encode Video Embedding
    input - clean video
    output - video encoding for clean video
    save as - .npz
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import argparse
from retrieveQuerySequences import formEmbedding
from imageCroppingClasses import imageCropping

useParser = 0
use3Channel = 1

frameListConcatenated = []
parser = argparse.ArgumentParser()
if useParser:
    parser.add_argument('-checkpointDir', type=str, default='./checkpoints/', help='enter checkpoint directory')
    parser.add_argument('-checkpointDir', type=str, default='./checkpoints/', help='enter checkpoint directory')
    args = parser.parse_args()
else:
    args = parser.parse_args()
    
    args.checkpointDir=os.path.join(ROOT_DIR, '/dysplasiaEndoscopy')
   
# check for existing encoding
embeddingFile= RESULT_DIR +'/'+'AE_'+patientFileName+'_ch_3.npy'
exists = os.path.isfile(embeddingFile)

if exists:
    print('embedding already exists, nothing to do!!!')
    loaded_embedding = 1
    x_train = ''
else:
    # todo: use the video reading function above here!!!!
    print('embedding doesnot exists, please wait while embedding is done!!!')
    nFramesSelected = [0, 5000]
    val_thresh = 30
    target_shape = (124,124)
    useGray=1
    
    #frameListConcatenated = videoFramesExtraction.extratedVideoFramesInArray(videoFile, nFramesSelected, useGray, target_shape, val_thresh )
    # grayscale image for now::: TODO change to color training
    if use3Channel:
        for i in range (0, len(cleanFrameList)):
            frameListConcatenated.append(np.reshape(cleanFrameList[i], (124,124,3)))
            
    else:
        for i in range (0, len(cleanFrameList)):
            frameListConcatenated.append(imageCropping.read_rgb(np.reshape(cleanFrameList[i], (124,124,3))  ))
        
    print('image files saved in an array, ''todo'' to write in a folder!!!')
    print(' ')
    
    ''' 
        Do embedding here 
    '''
    from keras.models import Model, load_model
    useDebug=0
    print('files being loaded (embedding set to 0 or not available)...\n...this will take a while...')
    
    ''' Start encoding here   '''
    t1 = time.time()
    x_train = np.reshape(frameListConcatenated, (len(frameListConcatenated), 124, 124, 3) )
    
    # BE_Autoencoder_124_124_ch3.h5, 33000_AE_1716_11012018-smallCNNFilters.h5
    autoencoder = load_model(os.path.join(ROOT_DIR, 'dysplasiaEndoscopy', 'BE_Autoencoder_124_124_ch3.h5'))
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
    # also saves as embedding file
    learned_codes = formEmbedding(encoder, x_train, embeddingFile, useDebug)
    t2 = time.time()
    
    print('Embedding done in',t2-t1, 'for video with frames: ', (nFramesSelected[1]-nFramesSelected[0]))
    print('summary of my network is:', autoencoder.summary())
    
    print('video cleaning done... saved encoded video asn npy, check:', RESULT_DIR)
    




