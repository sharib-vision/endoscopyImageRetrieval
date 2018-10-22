# @Author: Sharib Ali <shariba>
# @Date:   1970-01-01T00:00:00+00:00
# @Email:  sharib.ali@eng.ox.ac.uk
# @Project: BRC: VideoEndoscopy
# @Filename: train_model.py
# @Last modified by:   shariba
# @Last modified time: 2018-10-22T16:08:27+01:00
# @Copyright: 2018-2020, sharib ali

import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(3)

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.optimizers import RMSprop
import numpy as np


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
        if ext[0] in f or ext[1] in f:
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

def train_model(imageShape, totalEpochs, checkpointName, batchSize, x_train, x_train_noisy, x_test, x_test_noisy, checkpointDir):
    # TODO input size implicit
    input_img = Input(shape=(124, 124, 1))
    # adapt this if using `channels_first` image data format
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    print(autoencoder.summary())

    autoencoder.fit(x_train_noisy, x_train,
                epochs=totalEpochs,
                batch_size=batchSize,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir=checkpointDir, histogram_freq=0, write_graph=False)])

    autoencoder.save(checkpointName)


def train_model_filterSizeIncreased(imageShape, totalEpochs, checkpointName, batchSize):
    input_img = Input(shape=(124, 124, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

#    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    print(autoencoder.summary())
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=totalEpochs,
                    batch_size=batchSize,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

    autoencoder.save(checkpointName)


if __name__=="__main__":

    import argparse
    import os
    import numpy as np
    import pylab as plt
    import scipy.io as spio

    # parse the
    parser = argparse.ArgumentParser()
    parser.add_argument('-trainSet', action='store', help='filepath to training examples', type=str)
    parser.add_argument('-ValidationSet', action='store', help='filepath to validation examples', type=str)
    parser.add_argument('-quads', action='store', default=1, type=int, help='number to equaly divide image into')
    parser.add_argument('-epochs', action='store', default=20, type=int, help='number of epochs')
    parser.add_argument('-train_batch_size', type=int, default='1', help='enter training batch size')
    parser.add_argument('-shape', action='store', default=(124, 124), type=tuple, help='image size for training')
    parser.add_argument('-o',action='store', type=str, help='model name to save')
    parser.add_argument('-checkpointDir', type=str, default='./checkpoints/', help='enter checkpoint directory')
    args = parser.parse_args()

    ''' usage:

        python train_ae_.py -trainSet /well/rittscher/users/sharib/autoEncoder_BE/cycleGAN_WL_NBI_data/trainB/ \
        -ValidationSet /well/rittscher/users/sharib/autoEncoder_BE/BE_validation2/ -o BE_Autoencoder_WL_test.h5 \
        -epochs 3

    '''

    if not os.path.exists(args.checkpointDir):
        os.makedirs(args.checkpointDir)

    # load endoscopic Barrett's areas
    ext = ['.jpg', '.png']
    train_examples_folder = args.trainSet
    test_examples_folder = args.ValidationSet

    print('train samples:', train_examples_folder)
    print('validation samples:', test_examples_folder)
    print('train samples {} and validation samples {}'.format(len(os.listdir(train_examples_folder)), len(os.listdir(test_examples_folder))))

    train_files = load_imgs(train_examples_folder, ext)
    validation_files = load_imgs(test_examples_folder, ext)

    x_train_3 = read_images(train_files, shape=args.shape)
    x_test_3 = read_images(validation_files, shape=args.shape)

    x_train = np.reshape(x_train_3, (len(x_train_3), 124, 124, 1) )
    x_test = np.reshape(x_test_3, (len(x_test_3), 124, 124, 1) )

    # good to give noise_factor as this will deal with noisy images
    # TODO include to add other noise
    noise_factor = 0.25
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    batchSize = args.train_batch_size
    train_model(args.shape, args.epochs, args.checkpointDir+'/'+args.o, batchSize, x_train, x_train_noisy, x_test, x_test_noisy, args.checkpointDir)
