#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:21:05 2018

@author: shariba

retrieveQuerySequences.py
"""

import cv2
import time
import numpy as np

from skimage.transform import resize


def sortImages (encoder, useDebug, test_element, learned_codes, n_samples):
    
    test_code = encoder.predict(np.array([test_element]))
    test_code = test_code.reshape(test_code.shape[1] * test_code.shape[2] * test_code.shape[3])
    distances = []
    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
        
    listSorted=sorted(range(len(distances)),key=distances.__getitem__)
    kept_indexes = listSorted[:n_samples]
    if useDebug:
        print('soreted list:',listSorted)
        print('printing image files corresponding to-->', kept_indexes)
        
    return kept_indexes



def formEmbedding (encoder, x_train, embeddingFile, useDebug):
    t1 = time.time()
    learned_codes = encoder.predict(x_train)
    learned_codes = learned_codes.reshape(learned_codes.shape[0],
                                          learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])
    if useDebug:
        print(learned_codes)
        print(learned_codes.shape)
    
    np.save(embeddingFile, learned_codes)
    t2 = time.time()
    print('Autoencoder-Encoding done: ', t2-t1)
    return learned_codes


def retrieve_closest_images(test_element, test_label, n_samples, x_train, y_train, filenameToSave, fileListVideoSeq, 
                            loaded_embedding, embeddingFile, validation_files, index_test, useDebug, encoder, cpu):
    
    print('already learned embedding being loaded ... ')
    learned_codes = np.load(embeddingFile)
    usePreComputedCompression = 1
    
    if usePreComputedCompression:
        original_image = test_element
        
        textfilename = filenameToSave+'_retrievedImages_AE.txt'
        open(textfilename, 'w').close()
        textfile = open(textfilename, 'a')
        textfile.write('\n')
        
        kept_indexes = sortImages (encoder, 0, test_element, learned_codes, n_samples)

        
        if loaded_embedding:
            print('listing 100 sorted image files based on provided query sample ... ')
            for i in range(0, n_samples):
                j = kept_indexes[i]
                textfile.write(fileListVideoSeq[j])
                textfile.write('\n')
        else:
            if cpu:
                cv2.imshow('original_image', original_image)

            # Below codes are redundant
            retrieved_images = x_train[int(kept_indexes[0]), :]
            retrieved_images_1 = x_train[int(kept_indexes[0+n_samples]), :]

            for i in range(1, n_samples):
                retrieved_images = np.hstack((retrieved_images, x_train[int(kept_indexes[i]), :]))
            for i in range(1, n_samples):
                retrieved_images_1 = np.hstack((retrieved_images_1, x_train[int(kept_indexes[i+n_samples]), :]))
            # list 100 images in the text file to be given to siemese network for further sorting
            for i in range(0, 100):
                j = kept_indexes[i]
                textfile.write(fileListVideoSeq[j])
                textfile.write('\n')
            retrieved_final = np.vstack((retrieved_images, retrieved_images_1))
            if cpu:
                cv2.imshow('Results', retrieved_images)
                cv2.waitKey(0)

            cv2.imwrite('test_results/'+filenameToSave+'_original.jpg', 255 * cv2.resize(original_image, (0,0), fx=3, fy=3))
            cv2.imwrite('test_results/'+filenameToSave+'_retrieved.jpg', 255 * cv2.resize(retrieved_final, (0,0), fx=2, fy=2))
        textfile.close()
        useReadFromFile = 1
        n_samples=20

        if useReadFromFile:
            print('writing retrieved images for '+str(n_samples)+' samples...')
            shape=(124,124,3)
            #font = cv2.FONT_HERSHEY_PLAIN
    #        read original query file
            original_image = resize( (cv2.imread(validation_files[index_test],1)/255.), shape , mode='constant')
    #        read all images in the list
            dataList = open(textfilename, 'rt').read().split('\n')
            print(len(dataList))
            img = resize( (cv2.imread(dataList[1],1)/255.), shape , mode='constant')
    #        cv2.putText(img, '#'+str(kept_indexes[0]), (30, 10), font, 0.8, (0,255,0), 1 , cv2.LINE_AA)
            retrieved_images = img
    #            x_train[int(kept_indexes[0]), :]
    #        retrieved_images_1 = x_train[int(kept_indexes[0+n_samples]), :]
            for i in range (2, n_samples+1):
                print(dataList[i])
                img = resize( (cv2.imread(dataList[i],1)/255.), shape , mode='constant')
    #            cv2.putText(img, '#'+str(kept_indexes[i-1]), (30, 10), font, 0.8, (0,255,0), 1 , cv2.LINE_AA)
                retrieved_images = np.hstack((retrieved_images, img))

            cv2.imwrite(filenameToSave+'_retrieved_RGB.jpg', 255 * cv2.resize(retrieved_images, (0,0), fx=2, fy=2))
            cv2.imwrite(filenameToSave+'_original_RGB.jpg', 255*original_image)