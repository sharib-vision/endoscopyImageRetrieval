#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:27:05 2018

@author: shariba

"""

'''
this paths.py is for autoencoding of the videos 
'''


import os
import socket

hostname = socket.gethostname()
print(hostname)

localCopy = 1

if localCopy:
    ROOT_DIR = '/Users/shariba/dataset'
    BASE_DIR = '/Users/shariba/'
    DATA_DIR = os.path.join(ROOT_DIR, 'dysplasiaEndoscopy', 'videos')
    
else:
    if 'shariba' in hostname:
        ROOT_DIR = '/Volumes/rescomp2/data'
        BASE_DIR = '/Volumes/rescomp2/home'
        DATA_DIR = ROOT_DIR+'/videoData/mp4FilesForEncoding'
        
    else:
        ROOT_DIR = '/well/rittscher/users/sharib'
        BASE_DIR = '/users/rittscher/sharib'
        DATA_DIR = ROOT_DIR+'/videoData/mp4FilesForEncoding'
        

RESULT_DIR = ROOT_DIR + '/dysplasiaVideoEncoded'
os.makedirs(RESULT_DIR, exist_ok=True)

print('DATA_DIR: ', DATA_DIR)
print('RESULTS_DIR: ', RESULT_DIR)

