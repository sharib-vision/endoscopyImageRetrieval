#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:08:50 2018

@author: shariba
"""

def videoCleaningWithDNN (modelFile, videoFile):
    from keras.models import load_model
    from skimage.transform import resize
    from imageCroppingClasses import videoFramesExtraction, imageCropping
    import numpy as np
    
    # todo train with 124, 124
    target_shape = (124, 124)
    
    CNN_classify_model = load_model(modelFile)
    
    clip, (clip_fps, clip_duration, n_frames) = videoFramesExtraction.movieClipinfo(videoFile)    
    boundingbox = videoFramesExtraction.returnBoundingBox(videoFile, 30)    
    
    batch_size = 1
    # if batch is different than 1 this will not add up to the n_frames
    frame_scores = []
    cleanFrameList = []
    n_batches = int(np.ceil(n_frames/float(batch_size)))
    
    for i in range(n_batches)[:]:
        if i <n_batches:
            start = i*batch_size
            end = (i+1)*batch_size
            vid_frames = np.array([resize(imageCropping.crop_image( clip.get_frame(ii*1./clip_fps), boundingbox ), target_shape) for ii in range(start,end,1)]) 
        else:
            start = i*batch_size
            vid_frames = np.array([resize(imageCropping.crop_image( clip.get_frame(ii*1./clip_fps), boundingbox ), target_shape) for ii in range(start,n_frames,1)])       

        informativeness = CNN_classify_model.predict(vid_frames)
        information_index = np.argmax(informativeness, axis=1)
        
        frame_scores.append(information_index)
        if information_index == 1:
            # that is in our encoding
            cleanFrameList.append(vid_frames)
            
        
    frame_scores = np.hstack(frame_scores)
    
    return frame_scores, n_frames, cleanFrameList
