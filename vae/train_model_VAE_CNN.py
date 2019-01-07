import numpy as np

import matplotlib.pyplot as plt

from keras import backend as K
from keras import layers
from keras.models import Model
#from keras.datasets import mnist


cpu = 0

import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(3)


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

def read_rgb(f):
    
    import cv2
    img = cv2.imread(f,1)/255.
    #    im = [img[:, :, i].mean() for i in range(img.shape[-1])]
    #    print(im.shape)
    #    im = cv2.cvtColor(average_color,cv2.COLOR_BGR2GRAY )/ 255.
    # im = cv2.imread(f,0)
    # im2 = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    [b,g,r] = cv2.split(img)
    #    luminosity method accounting for human perception
    im = (0.07*b+0.72*g+0.21*r)
    
    #    im = (0.07*b+0.72*g+0.21*r)/3
    return im


def read_images(flist, shape):

    from skimage.transform import resize

    imgs = []

    for f in flist:
        im = read_rgb(f)
        im = resize(im, shape, mode='constant')
        imgs.append(im[None,:])

    return np.concatenate(imgs, axis=0)


# Dimensions of MNIST images
#image_shape = (28, 28, 1)

# Dimensions of endoscopy images
image_shape = (124, 124, 1)
half_size=int(0.5*image_shape[0])
unflatten = int(half_size * half_size * 64)
# Dimension of latent space
latent_dim = 2

# Mini-batch size for training
batch_size = 64

def create_encoder():
    '''
        Creates a convolutional encoder model for MNIST images.
        
        - Input for the created model are MNIST images.
        - Output of the created model are the sufficient statistics
        of the variational distriution q(t|x;phi), mean and log
        variance.
        '''
    encoder_iput = layers.Input(shape=image_shape)
    
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(encoder_iput)
#    downscaled by 2 here
    x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    #    14 x 14 x 64 = 12544
    x = layers.Flatten()(x)

    # only keep 32 dense connections     
    x = layers.Dense(32, activation='relu')(x)
    
    t_mean = layers.Dense(latent_dim)(x)
    t_log_var = layers.Dense(latent_dim)(x)
    
    return Model(encoder_iput, [t_mean, t_log_var], name='encoder')

def create_decoder():
    '''
        Creates a (de-)convolutional decoder model for MNIST images.
        
        - Input for the created model are latent vectors t.
        - Output of the model are images of shape (28, 28, 1) where
        the value of each pixel is the probability of being white.
        '''
    decoder_input = layers.Input(shape=(latent_dim,))

    # 14 x 14 x64
#    x = layers.Dense(12544, activation='relu')(decoder_input)
#    (0.5*124)**2*64
    x = layers.Dense(unflatten, activation='relu')(decoder_input)
    x = layers.Reshape((half_size, half_size, 64))(x)
#    x = layers.Reshape((14, 14, 64))(x)
#    upscale by 2 (28x28-->32 filters)
    x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    
    return Model(decoder_input, x, name='decoder')


def sample(args):
    '''
        Draws samples from a standard normal and scales the samples with
        standard deviation of the variational distribution and shifts them
        by the mean.
        
        Args:
        args: sufficient statistics of the variational distribution.
        
        Returns:
        Samples from the variational distribution.
        '''
    t_mean, t_log_var = args
    t_sigma = K.sqrt(K.exp(t_log_var))
    epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
    return t_mean + t_sigma * epsilon

def create_sampler():
    '''
        Creates a sampling layer.
        '''
    return layers.Lambda(sample, name='sampler')


encoder = create_encoder()
decoder = create_decoder()
sampler = create_sampler()

x = layers.Input(shape=image_shape)
t_mean, t_log_var = encoder(x)
t = sampler([t_mean, t_log_var])
t_decoded = decoder(t)

vae = Model(x, t_decoded, name='vae')


def neg_variational_lower_bound(x, t_decoded):
    '''
        Negative variational lower bound used as loss function
        for training the variational auto-encoder.
        
        Args:
        x: input images
        t_decoded: reconstructed images
        '''
    # Reconstruction loss
    rc_loss = K.sum(K.binary_crossentropy(K.batch_flatten(x), K.batch_flatten(t_decoded)), axis=-1)
        
# Regularization term (KL divergence)
    kl_loss = -0.5 * K.sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=-1)
                                          
                                          # Average over mini-batch
    return K.mean(rc_loss + kl_loss)


if __name__=="__main__":

    import argparse
    import os

    # parse the
    parser = argparse.ArgumentParser()
    parser.add_argument('-trainSet', action='store', help='filepath to training examples', type=str)
    parser.add_argument('-ValidationSet', action='store', help='filepath to validation examples', type=str)
    parser.add_argument('-quads', action='store', default=1, type=int,
                        help='number to equaly divide image into')
    parser.add_argument('-split', action='store', default=0.6, type=float, help='train-test split')
    parser.add_argument('-epochs', action='store', default=2, type=int, help='number of epochs')
    parser.add_argument('-shape', action='store', default=(124,124), type=tuple, help='image size for training')
    parser.add_argument('-o',action='store', type=str, help='model name to save')
    parser.add_argument("-w", "--weights", help='load previous weights possible')
    parser.add_argument("-m", "--mse", help='default is cross-entropy')
    args = parser.parse_args()
    
    '''
    RUN: python train_model_VAE_CNN.py -trainSet /well/rittscher/users/sharib/autoEncoder_BE/BE_Area/ \
    -ValidationSet /well/rittscher/users/sharib/autoEncoder_BE/BE_validation2/ -o BE_VAE_Autoencoder_WL-CNN.h5 -epochs 3

    '''
    
    
    '''
    
    TODO: change shapes and measure accuracy of VAEs
    VAE size  === AE size and compare compression and accuracy
    
    '''    
    # load endoscopic Barrett's areas
    ext = ['.jpg', '.png']
    train_examples_folder = args.trainSet
    test_examples_folder = args.ValidationSet
#    nbi_examples_folder = args.trainSet

    print('train samples:', train_examples_folder)
    print('validation samples:', test_examples_folder)
#    print('NBI samples:', nbi_examples_folder)

    train_files = load_imgs(train_examples_folder, ext)
    validation_files = load_imgs(test_examples_folder, ext)

    x_train_3 = read_images(train_files, shape=args.shape)
    x_test_3 = read_images(validation_files, shape=args.shape)
    
    x_train = np.reshape(x_train_3, (len(x_train_3), 124, 124, 1) )
    x_test = np.reshape(x_test_3, (len(x_test_3), 124, 124, 1) )
    
    
    # Compile variational auto-encoder model
    vae.compile(optimizer='rmsprop', loss=neg_variational_lower_bound)
    
    # Train variational auto-encoder with endoscopic images
    vae.fit(x=x_train,
            y=x_train,
            epochs=args.epochs,
            shuffle=True,
            batch_size=batch_size,
            validation_data=(x_test, x_test), verbose=2)
    
    vae.save(args.o)
    encoder.save("encoder_VAE_CNN.h5")
    decoder.save("decoder_VAE_CNN.h5")
    
    
    
    if cpu:
        
        # Generate latent vectors of validation set
        t_test = encoder.predict(x_test)[0]
        
        # Plot latent vectors colored by the value of the digits on input images
        y_test=['red', 'green', 'cyan', 'magenta', 'yellow', 'gray', 'pink']
        plt.scatter(t_test[:, 0], t_test[:, 1], marker='x', s=0.2, c=y_test)
        plt.colorbar();
        
        
        from scipy.stats import norm
        
        # Number of samples per dimension
        n = 15
        
        # Sample within 90% confidence interval of the Gaussian prior
        # with sampling density proportional to probability density
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                t_sample = np.array([[xi, yi]])
                t_sample = np.tile(t_sample, batch_size).reshape(batch_size, 2)
                t_decoded = decoder.predict(t_sample, batch_size=batch_size)
                digit = t_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
        
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r');
        
