import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import keras
from keras import backend as K
from keras import layers
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.utils import to_categorical

# Dimensions of MNIST images
image_shape = (28, 28, 1)

# Dimension of latent space
latent_dim = 2

# Mini-batch size for training
batch_size = 128

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
    x = layers.Dense(12544, activation='relu')(decoder_input)
    x = layers.Reshape((14, 14, 64))(x)
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


# MNIST training and validation data
(x_train, _), (x_test, y_test) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))

# Compile variational auto-encoder model
vae.compile(optimizer='rmsprop', loss=neg_variational_lower_bound)

# Train variational auto-encoder with MNIST images
vae.fit(x=x_train,
        y=x_train,
        epochs=25,
        shuffle=True,
        batch_size=batch_size,
        validation_data=(x_test, x_test), verbose=2)


# Generate latent vectors of validation set
t_test = encoder.predict(x_test)[0]

# Plot latent vectors colored by the value of the digits on input images
#y_test=['red', 'green', 'cyan', 'magenta']
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

