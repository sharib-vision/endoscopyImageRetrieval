#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 11:54:46 2018

@author: shariba
"""

'''
    imageCroppingClasses.py
    this python script consists of class definations related to image cropping

'''

class imageProperties:

    def detect_imgs(infolder, ext='.tif'):
        import os
        import numpy as np
        items = os.listdir(infolder)
        flist = []
        for names in items:
            if names.endswith(ext) or names.endswith(ext.upper()):
                flist.append(os.path.join(infolder, names))
        return np.sort(flist)

class imageCropping:
    
    def read_rgb(img):
        import cv2
        [b,g,r] = cv2.split(img)
        im = (0.07*b+0.72*g+0.21*r)
        return im
    
    def getLargestBBoxArea (img, val_thresh):
        import cv2
        import numpy as np
        
        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        ret,thresh = cv2.threshold(imgray, val_thresh, 255, 0)  
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # find the maximum contour area
        areaCnt = []
        for i in range (0, len(contours)):
            areaCnt.append(cv2.contourArea(contours[i]))
        largest_region=np.argmax(areaCnt)
        boundingBox = cv2.boundingRect(contours[largest_region])
        
        return boundingBox
        
    def crop_image( img, mask_bbox):
        (x_min, y_min, x_max, y_max) = mask_bbox
        return img[y_min:y_max+y_min, x_min:x_max+x_min]
    
    def draw_bounding_box(img, bbox,thickness):
        import cv2
        (x_min, y_min, x_max, y_max) = bbox
        boxed_img = cv2.rectangle(img, (x_min,y_min),(x_max+x_min,y_max+y_min),(0,155,0),thickness)
        return boxed_img


class videoFramesExtraction:
    
    def movieClipinfo(testvideofile):
        from moviepy.editor import VideoFileClip
        #from moviepy.editor import *
        clip = VideoFileClip(testvideofile)
         # extract the meta information from the file.
        clip_fps = clip.fps
        clip_duration = clip.duration
        n_frames = int(clip.duration/(1./clip_fps))
        return clip, [clip_fps, clip_duration, n_frames]
    
    
    def extratedVideoFramesInArray(videoFile, nFramesSelected, useGray, target_shape, val_thresh):
        from skimage.transform import resize
        import numpy as np
        clip, (clip_fps, clip_duration, n_frames) = videoFramesExtraction.movieClipinfo(videoFile)
        frame0 = clip.get_frame(0*clip_fps) # use the first frame to clip.
        #get mask
        boundingBox = imageCropping.getLargestBBoxArea (frame0, val_thresh)
        img_cropped = imageCropping.crop_image(frame0, boundingBox)
        imageDebugging.showImageDebug(img_cropped)
        famesListed=[]
        for i in range(nFramesSelected[0], nFramesSelected[1]):
            img = clip.get_frame(np.ceil(i))
            if useGray:
                img_gray= imageCropping.read_rgb(img)
            else:
                img_gray= img
            image_cropped=imageCropping.crop_image(img_gray, boundingBox)
        #    imageDebugging.showImageDebug(image_cropped)
            famesListed.append([resize(image_cropped, target_shape)])
        frameListConcatenated=np.concatenate(famesListed, axis=0) 
        return frameListConcatenated
    
    def returnBoundingBox(videoFile, val_thresh):
        clip, (clip_fps, clip_duration, n_frames) = videoFramesExtraction.movieClipinfo(videoFile)
        frame0 = clip.get_frame(0*clip_fps) # use the first frame to clip.
        #get mask
        boundingBox = imageCropping.getLargestBBoxArea (frame0, val_thresh)
        return boundingBox
    
#    def convertFramestoVideo (videoFile, frame_score, cleanVideoFileName):
#        import cv2
#        
#        for image in images:
#            video.write(cv2.imread(os.path.join(image_folder, image)))
#    
#        video = cv2.VideoWriter(cleanVideoFileName, -1, 1, (width,height))
#        cv2.destroyAllWindows()
#        video.release()
#
#        return x
#         

        
    

class imageDebugging:
    def showImageDebug(img):
        import matplotlib.pyplot as plt 
        plt.imshow(img)
        plt.show()
    def showImageDebug_matchChannels(img):
        import matplotlib.pyplot as plt 
        plt.imshow(img[:,:,[2,1,0]])
        plt.show()