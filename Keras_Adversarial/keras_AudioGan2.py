import sys
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,RMSprop
from keras.initializations import normal
import numpy as np
from scipy.stats import gaussian_kde
from scipy.io import wavfile
import theano.tensor as T
import theano
import pydub

batch_size = 32

print "loading data"

f = pydub.AudioSegment.from_mp3('./amclassical_beethoven_fur_elise.mp3')
data = np.fromstring(f._data, np.int16)
data = data.astype(np.float64).reshape((-1,2))
print data.shape
data = data[:,0]+data[:,1]
#data = data[:,:subsample*int(len(data)/subsample)-1,:]
data -= data.min()
data /= data.max() / 2.
data -= 1.
print data.shape

print "Setting up decoder"
decoder = Sequential()
decoder.add(Convolution1D(4, 256, subsample_length = 4, input_shape=(32768,1), activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Convolution1D(16, 256, input_dim=4, subsample_length = 16, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Flatten())
decoder.add(Dense(1, activation='sigmoid'))
decoder.summary()

#sgd = SGD(lr=0.01, momentum=0.1)
sgd1 = RMSprop()
decoder.compile(loss='binary_crossentropy', optimizer=sgd1)

print "Setting up generator"
generator = Sequential()
generator.add(Convolution1D(16, 64, activation='relu', input_shape=(2048,16),border_mode='same'))
generator.add(BatchNormalization())
generator.add(UpSampling1D(4))
generator.add(Convolution1D(4, 256, activation='relu',border_mode='same'))
generator.add(BatchNormalization())
generator.add(UpSampling1D(4))
#(((2048+64-1)*4 + 256 - 1)*4 - 2029 + 1)
generator.add(Convolution1D(1, 512, activation='linear',border_mode='same'))
generator.summary()

sgd = RMSprop()
generator.compile(loss='binary_crossentropy', optimizer=sgd)

print "Setting up combined net"
gen_dec = Sequential()
gen_dec.add(generator)
decoder.trainable=False
gen_dec.add(decoder)
sgd3 = RMSprop()

gen_dec.compile(loss='binary_crossentropy', optimizer=sgd3)

y_decode = np.ones(2*batch_size)
y_decode[:batch_size] = 0.
y_gen_dec = np.ones(batch_size)

def gaussian_likelihood(X, u=0., s=1.):
    return (1./(s*np.sqrt(2*np.pi)))*np.exp(-(((X - u)**2)/(2*s**2)))


for i in range(1000000):
    print " ", i
    zmb = np.random.normal(0., 1, size=(batch_size, 2048, 16)).astype('float32')
    #xmb = np.random.normal(1., 1, size=(batch_size, 1)).astype('float32')
    xmb = np.array([data[n:n+32768] for n in np.random.randint(0,data.shape[0]-32768,batch_size)]).astype('float32')
    xmb = xmb[:,:,np.newaxis]
    if i % 10 == 0:
        err_E = 2
        while err_E > 0.9:
            r = gen_dec.fit(zmb,y_gen_dec,nb_epoch=1,verbose=0)
            err_E = r.history['loss'][0]/batch_size
            print 'E:',err_E
    else:
        r = decoder.fit(np.vstack([generator.predict(zmb),xmb]),y_decode,nb_epoch=1,verbose=0)
        print 'D:',(r.history['loss'][0]/batch_size)
    if i % 100 == 0:
        print "saving fakes"
        fakes = generator.predict(zmb[:16,:])
        wavfile.write('fake_epoch_'+str(i)+'.wav',44100,fakes[0,:])
        for n in range(16):
            wavfile.write('fake_'+str(n+1)+'.wav',44100,fakes[n,:])
            wavfile.write('real_'+str(n+1)+'.wav',44100,xmb[n,:])
#        vis(i)