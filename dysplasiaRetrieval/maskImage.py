#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:21:10 2018

@author: shariba
"""

import os
import numpy as np
from imageCroppingClasses import imageCropping  
    
def detect_imgs(infolder, ext='.tif'):
    
    items = os.listdir(infolder)
    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))
    return np.sort(flist)

if __name__=="__main__":
    
    import cv2
    import matplotlib.pyplot as plt 
    useDebug = 0
    # TODO: use parser
    filenameSaveFolder =  '/Users/shariba/dataset/dysplasiaEndoscopy/croppedImages'
    os.makedirs(filenameSaveFolder, exist_ok=True)
    
    '''list images in your folder'''
    folderName='/Users/shariba/dataset/dysplasiaEndoscopy/images'
    listImage = detect_imgs(folderName, '.bmp')
    val_thresh = 30
    
    for k in range (0, len(listImage)):
        # get binary
        img = cv2.imread(listImage[k])
        file = listImage[k].split('/')[6]    #hardcoded
        if k ==9:           # the image is very dark
            val_thresh = 10
        boundingBox = imageCropping.getLargestBBoxArea (img, val_thresh)
        img_cropped = imageCropping.crop_image(img, boundingBox)
        
        # display
        if useDebug:
            img_cropped = imageCropping.crop_image(img[:,:,[2,1,0]], boundingBox)
            plt.imshow(img_cropped)
            plt.show()
        else:
            cv2.imwrite(filenameSaveFolder+'/'+file, img_cropped)
