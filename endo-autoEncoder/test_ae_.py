# @Author: Sharib Ali <shariba>
# @Date:   1970-01-01T00:00:00+00:00
# @Email:  sharib.ali@eng.ox.ac.uk
# @Project: BRC: VideoEndoscopy
# @Filename: test_model.py
# @Last modified by:   shariba
# @Last modified time: 2018-10-22T16:30:10+01:00
# @Copyright: 2018-2020, sharib ali


# with embedding load time: 3.888274908065796s for 20k images (3.1GB-->250.6 MB)
# ----> 708.156613111496s (without embedding)

import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

import numpy as np
from keras.models import Model
from keras.datasets import mnist
import cv2
from keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score
from skimage.transform import resize
import time

useDebug = 1

def detect_imgs(infolder, ext='.tif'):

    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)

def load_imgs( infolder, ext ):

    import os

    files = os.listdir(infolder)
    img_files = []

    for f in files:
        if ext in f:
            img_files.append(os.path.join(infolder, f))

    return np.hstack(img_files)

def read_images(flist, shape):

    from skimage.transform import resize

    imgs = []

    for f in flist:
        im = read_rgb(f)
        im = resize(im, shape, mode='constant')
        imgs.append(im[None,:])

    return np.concatenate(imgs, axis=0)

def read_rgb_gray(f):
   import cv2
   im = cv2.imread(f,0) / 255.
   return im

def read_rgb(f):
    import cv2
    img = cv2.imread(f,1)/255.
    [b,g,r] = cv2.split(img)
    im = (0.07*b+0.72*g+0.21*r)
    return im

def retrieve_closest_elements(test_code, test_label, learned_codes):
    distances = []
    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)
        distances.append(distance)
    nb_elements = learned_codes.shape[0]
    distances = np.array(distances)
    learned_code_index = np.arange(nb_elements)
    labels = np.copy(y_train).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == -1] = 0
    distance_with_labels = np.stack((distances, labels, learned_code_index), axis=-1)
    sorted_distance_with_labels = distance_with_labels[distance_with_labels[:, 0].argsort()]

    sorted_distances = 28 - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    return sorted_distances, sorted_labels, sorted_indexes


def compute_average_precision_score(test_codes, test_labels, learned_codes, n_samples):
    out_labels = []
    out_distances = []
    retrieved_elements_indexes = []
    for i in range(len(test_codes)):
        sorted_distances, sorted_labels, sorted_indexes = retrieve_closest_elements(test_codes[i], test_labels[i], learned_codes)
        out_distances.append(sorted_distances[:n_samples])
        out_labels.append(sorted_labels[:n_samples])
        retrieved_elements_indexes.append(sorted_indexes[:n_samples])

    out_labels = np.array(out_labels)
    out_labels_file_name = 'computed_data/out_labels_{}'.format(n_samples)
    np.save(out_labels_file_name, out_labels)

    out_distances_file_name = 'computed_data/out_distances_{}'.format(n_samples)
    out_distances = np.array(out_distances)
    np.save(out_distances_file_name, out_distances)
    score = label_ranking_average_precision_score(out_labels, out_distances)
    scores.append(score)
    return score


def retrieve_closest_images(test_element, test_label, n_samples, x_train, y_train, filenameToSave, fileListVideoSeq, loaded_embedding, embeddingFile, validation_files, index_test):

    if loaded_embedding:
        print('already learned embedding being loaded ... ')
        learned_codes = np.load(embeddingFile)
    else:
        # One do for train data only if the compression is not available
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

    usePreComputedCompression = 1
    if usePreComputedCompression:
        test_code = encoder.predict(np.array([test_element]))
        test_code = test_code.reshape(test_code.shape[1] * test_code.shape[2] * test_code.shape[3])
        distances = []

        textfilename = filenameToSave+'_retrievedImages_AE.txt'
        open(textfilename, 'w').close()
        textfile = open(textfilename, 'a')
        textfile.write('\n')

        for code in learned_codes:
            distance = np.linalg.norm(code - test_code)
            distances.append(distance)

        listSorted=sorted(range(len(distances)),key=distances.__getitem__)
        kept_indexes = listSorted[:n_samples]

        if useDebug:
            print('soreted list:',listSorted)
            print('printing image files corresponding to-->', kept_indexes)

        original_image = test_element

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
            font = cv2.FONT_HERSHEY_PLAIN
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


def test_model(n_test_samples, n_train_samples, y_test):
    learned_codes = encoder.predict(x_train)
    learned_codes = learned_codes.reshape(learned_codes.shape[0], learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])
    # query encoding
    test_codes = encoder.predict(x_test)
    test_codes = test_codes.reshape(test_codes.shape[0], test_codes.shape[1] * test_codes.shape[2] * test_codes.shape[3])
    indexes = np.arange(len(y_test))
    np.random.shuffle(indexes)
    indexes = indexes[:n_test_samples]
    print('Start computing score for {} train samples'.format(n_train_samples))
    t1 = time.time()
    score = compute_average_precision_score(test_codes[indexes], y_test[indexes], learned_codes, n_train_samples)
    t2 = time.time()
    print('Score computed in: ', t2-t1)
    print('Model score:', score)


def plot_denoised_images():
    denoised_images = autoencoder.predict(x_test_noisy.reshape(x_test_noisy.shape[0], x_test_noisy.shape[1], x_test_noisy.shape[2], 1))
    test_img = x_test_noisy[0]
    resized_test_img = cv2.resize(test_img, (280, 280))
    cv2.imshow('input', resized_test_img)
    cv2.waitKey(0)
    output = denoised_images[0]
    resized_output = cv2.resize(output, (280, 280))
    cv2.imshow('output', resized_output)
    cv2.waitKey(0)
    cv2.imwrite('test_results/noisy_image.jpg', 255 * resized_test_img)
    cv2.imwrite('test_results/denoised_image.jpg', 255 * resized_output)



if __name__=="__main__":

    import argparse
    import os
    import numpy as np
    import pylab as plt
    import scipy.io as spio

    parser = argparse.ArgumentParser()
    parser.add_argument('-trainSet', action='store', help='filepath to video frames for compression', type=str)
    parser.add_argument('-ValidationSet', action='store', help='filepath to query images', type=str)
    parser.add_argument('-resultDir', action='store', help='filepath to results folder', type=str)
    parser.add_argument('-patientName', action='store', help='patient ID (usually folder name of the validationSet or anything folder name corresponding to patient)', type=str)
    parser.add_argument('-shape', action='store', default=(124,124), type=tuple, help='image size for training')
    parser.add_argument('-checkpointDir', type=str, default='./checkpoints/', help='enter checkpoint directory')
    parser.add_argument('-o',action='store', type=str, help='model name to save')
    args = parser.parse_args()


    ''' usage:

        python test_ae_.py -trainSet /well/rittscher/projects/endoscopy_ALI/20042017120301/ \
        -ValidationSet /well/rittscher/projects/endoscopy_ALI/test/ -o ./checkpoints/BE_Autoencoder_WL_test.h5 -resultDir 20042017120301_test

    '''

    if not os.path.exists(args.resultDir):
        os.makedirs(args.resultDir)

    print('Loading images...')
    t0 = time.time()
    print('Loading model :')
    autoencoder = load_model(args.o)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
    t1 = time.time()
    print('Model loaded in: ', t1-t0)
    t0 = time.time()

    # Flags
    ''' set loaded_embedding = 1 if embedding is present else 0'''
    WL = 1
    cpu = 0
    loaded_embedding = 0

     # load endoscopic Barrett's areas
    print('Loading Barrets dataset')
    ext = '.jpg'
    train_examples_folder = args.trainSet
    test_examples_folder = args.ValidationSet

    print('train samples:', train_examples_folder)
    print('validation samples:', test_examples_folder)
    print('encoding/train samples {} and query samples {}'.format(len(os.listdir(train_examples_folder)), len(os.listdir(test_examples_folder))))

    # store sorted images here
    train_files = detect_imgs(train_examples_folder, ext)
    validation_files = detect_imgs(test_examples_folder, ext)

    # 'AE_'+patientFileName+'_WL_3.npy'
    patientFileName=args.patientName

    if loaded_embedding:
        print('files being loaded as embedding...')
        x_train = ''
    else :
        exists = os.path.isfile(args.checkpointDir+'/'+'AE_'+patientFileName+'_WL_3.npy')
        if exists:
            print('embedding exists and loading...')
            loaded_embedding = 1
            x_train = ''
        else:
            print('files being loaded (embedding set to 0 or not available)...\n...it will take a while...')
            t1 = time.time()
            x_train_3 = read_images(train_files, shape=args.shape)
            x_train = np.reshape(x_train_3, (len(x_train_3), 124, 124, 1) )
            t2 = time.time()
            print('Loading training data done in: ', t2-t1)

    x_test_3 = read_images(validation_files, shape=args.shape)
    x_test = np.reshape(x_test_3, (len(x_test_3), 124, 124, 1) )

    if loaded_embedding:
        y_train = ' '
    else:
        y_train = x_train

    y_test = x_test

    t1 = time.time()
    print('Training data loaded in: ', t1-t0)

    scores = []

    filenameToSave=''
    embeddingFile=''
    index_test = ''

    # To retrieve closest image
    # TODO: list images in a file and argument for saving retrieved images to a folder
    #    filenameToSave = 'query_2_AE_NBI_20042017120301'
    #    patientFileName='17082017114319'
    #    patientFileName='20042017120301'
    #    1716 Patient:2

    #    patientFileName='26042018085723_1716'
    #    patientFileName='kvasir'

    #    patientFileName='20042017120301_AE_trained_30000'
    #TIME

    filenameToSave =args.resultDir+'/'+'query_'+patientFileName
    textfilename_queryList = filenameToSave+'_queryList.txt'
    open(textfilename_queryList, 'w').close()
    textfileQuery = open(textfilename_queryList, 'a')
    textfileQuery.write('\n')

    # choose loading pre-existing embedding?
    #    embeddingFile = "AE_20042017120301_NBI.npy"

    for k in range(1, len(os.listdir(test_examples_folder))):
        index_test = k
        textfileQuery.write(validation_files[index_test])
        t0 = time.time()
        if WL:
            filenameToSave = args.resultDir+'/'+'query_'+str(index_test)+'_AE_WL_'+patientFileName
            embeddingFile = args.checkpointDir+'/'+'AE_'+patientFileName+'_WL_3.npy'

            textfilename_queryList = filenameToSave+'_queryList.txt'
            open(textfilename_queryList, 'w').close()
            textfileQuery = open(textfilename_queryList, 'a')
            textfileQuery.write('\n')

        else:
            filenameToSave = args.resultDir+'/'+'query_'+str(index_test)+'_AE_NBI_'+patientFileName
            embeddingFile = args.checkpointDir+'/'+'AE_'+patientFileName+'_NBI_3.npy'

        retrieve_closest_images(x_test[index_test], y_test, 100, x_train, y_train, filenameToSave, train_files, loaded_embedding, embeddingFile,validation_files, index_test)
        t1 = time.time()
        print('Retrived image in: ', t1-t0)
