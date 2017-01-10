#
# Adapted from :
#
#  http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

import os

import numpy as np

from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

import matplotlib.pyplot as pyplot

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

FTRAIN = './data/training-facial.csv'
FTEST = './data/test-facial.csv'
# network and training
N_HIDDEN = 512
NB_EPOCH = 10
BATCH_SIZE = 32
VERBOSE = 1
VALIDATION_SPLIT=0.2

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

#
# reshape (1, 96, 96) so that this is ready for convnet on 1 color (gray)
#
def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def simpleNet(X_train, Y_train):
	model = Sequential()

	# input shape is (variable size x 9216 = 96x96)
	# hidden layer is 100 units
	model.add(Dense(100, input_shape=(9216,)))
	model.add(Activation('relu'))
	model.add(Dense(30))
	# If you don't specify anything, no activation is applied 
	# (ie. "linear" activation: a(x) = x).
	model.summary()

	# this is then seen as a regression problem
	# see https://github.com/fchollet/keras/issues/108
	model.compile(loss='mean_absolute_error', optimizer='rmsprop')
	
	history = model.fit(X_train, Y_train,
                    nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

	return model

#
# A more sophisticate net using convolution
#
def convNet(X, y):
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, 
		input_shape=(1, 96, 96,)))
	model.add(Activation('relu'))
	# conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2)
	model.add(Convolution2D(32, 3, 3))  
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.1))
	# conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2)
	model.add(Convolution2D(64, 2, 2))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	# conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
	model.add(Convolution2D(128, 2, 2))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))

	model.add(Flatten())

	# hidden4_num_units=500
	model.add(Dense(500))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	# hidden5_num_units=500
	model.add(Dense(500))
	model.add(Activation('relu'))
	# output 
	model.add(Dense(30))
	# output_nonlinearity=None
	# If you don't specify anything, no activation is applied 
	# (ie. "linear" activation: a(x) = x).

	model.summary()

	# this is then seen as a regression problem
	# see https://github.com/fchollet/keras/issues/108
	model.compile(loss='mean_absolute_error', optimizer='rmsprop')
	
	history = model.fit(X, y,
                    nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

	return model
#
# annotate image with keypoints
#
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


# run first net
if (0):
	X, y = load()
	print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
	    X.shape, X.min(), X.max()))
	print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
	    y.shape, y.min(), y.max()))
	net1 = simpleNet(X, y)

	X, _ = load(test=True)
	y_pred = net1.predict(X)

	print ('prediction done')

	# show the images
	fig = pyplot.figure(figsize=(6, 6))
	fig.subplots_adjust(
	    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
	    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
	    plot_sample(X[i], y_pred[i], ax)

	pyplot.show()

else:
# run second net
	X, y = load2d()
	print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
	    X.shape, X.min(), X.max()))
	print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
	    y.shape, y.min(), y.max()))
	net2 = convNet(X, y)