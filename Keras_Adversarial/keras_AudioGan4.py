import sys
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD,RMSprop,Adam
from keras.initializations import normal
import numpy as np
from scipy.stats import gaussian_kde
from scipy.io import wavfile
import theano.tensor as T
import theano
import pydub

import matplotlib.pyplot as plt
plt.ion()

batch_size = 32

print "loading data"

f = pydub.AudioSegment.from_mp3('./amclassical_beethoven_fur_elise.mp3')
data = np.fromstring(f._data, np.int16)

#why this?
#print data.shape
data = data.astype(np.float64)
#data = data.astype(np.float64).reshape((-1,2))
#print data.shape
#data = data[:,0]+data[:,1]
#data = data[:,:subsample*int(len(data)/subsample)-1,:]

#normalizaition
data -= data.min()
data /= data.max() / 2.
data -= 1.
#normalize the images between -1 and 1
print data.shape

#LeakyReLU = good (in both G an D)
print "Setting up discriminator"
discriminator = Sequential()
discriminator.add(Convolution1D(4, 256, subsample_length = 4, input_shape=(32768,1)))
discriminator.add(LeakyReLU())
#Use Dropouts in G in both train and test phase
discriminator.add(Dropout(0.5))
discriminator.add(Convolution1D(16, 256, input_dim=4, subsample_length = 16))
discriminator.add(LeakyReLU())
discriminator.add(Dropout(0.5))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()
#Use SGD for discriminator and ADAM for generator
sgd1 = SGD()
discriminator.compile(loss='binary_crossentropy', optimizer=sgd1)

print "Setting up generator"
generator = Sequential()
generator.add(Convolution1D(16, 64, input_shape=(2048,16),border_mode='same'))
generator.add(LeakyReLU())
#Use Dropouts in G in both train and test phase
generator.add(Dropout(0.5))
generator.add(BatchNormalization())
generator.add(UpSampling1D(4))
generator.add(Convolution1D(64, 256,border_mode='same'))
generator.add(LeakyReLU())
#Use Dropouts in G in both train and test phase
generator.add(Dropout(0.5))
generator.add(BatchNormalization())
generator.add(UpSampling1D(4))
#Tanh as the last layer of the generator output
generator.add(Convolution1D(1, 512, activation='tanh',border_mode='same'))
generator.summary()
#Use SGD for discriminator and ADAM for generator
sgd = Adam()
generator.compile(loss='binary_crossentropy', optimizer=sgd)

print "Setting up combined net"
gen_dec = Sequential()
gen_dec.add(generator)
discriminator.trainable=False
gen_dec.add(discriminator)
sgd3 = Adam()
gen_dec.compile(loss='binary_crossentropy', optimizer=sgd3)

# a vector of size batch_size, full of 1 
y_discriminator = np.ones(2*batch_size)
# half of it is now full of 0
# [000000.. 111111]
# [forged.. real..]
y_discriminator[:batch_size] = 0.   

# the gen_discr is full of 1
y_gen_discr = np.ones(batch_size)
#[111111..]
#[forged..]

generator_loss = []
discriminator_loss = []

for i in range(1000000):
    print " ", i

    # generate noise with gaussian mean=0, dev=1, sample of size batch_size * 2048 * 16
    zmb = np.random.normal(0., 1, size=(batch_size, 2048, 16)).astype('float32')
#    print 'shape of zmb', zmb.shape
#    print "shape of label", y_gen_discr.shape

    # sample the real data
    xmb = np.array([data[n:n+32768] for n in np.random.randint(0,data.shape[0]-32768,batch_size)]).astype('float32')
    xmb = xmb[:,:,np.newaxis]
#   print "shape of xmb", xmb.shape
#   print "shape of label", y_discriminator.shape

    subiteration = 0
    while 1:    
        r = gen_dec.fit(zmb,y_gen_discr,nb_epoch=1,verbose=1)
        loss_G = r.history['loss'][0]
        generator_loss.append(loss_G)
        print 'G:', loss_G
        subiteration +=1 
        if (loss_G < 2 and subiteration < 5):
            break

    subiteration = 0
    while 1:
        r = discriminator.fit(np.vstack([generator.predict(zmb),xmb]),y_discriminator,nb_epoch=1,verbose=1)
        loss_D = r.history['loss'][0]
        discriminator_loss.append(loss_D)
        print 'D:', loss_D     
        subiteration += 1
        if (loss_D < 2  and subiteration < 5):
            break

    if i % 2 == 0:
        plt.plot(generator_loss)
        plt.plot(discriminator_loss)
        plt.draw()
        plt.pause(0.001)

    if i % 5 == 0:    
        print "saving fakes"
        fakes = generator.predict(zmb[:16,:])
        wavfile.write('fake_epoch_'+str(i)+'.wav',44100,fakes[0,:])
        for n in range(16):
            wavfile.write('fake_'+str(n+1)+'.wav',44100,fakes[n,:])
            wavfile.write('real_'+str(n+1)+'.wav',44100,xmb[n,:])
       
#        vis(i)