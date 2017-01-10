#
# working
# 
#
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils, generic_utils

# Set constants
BATCH_SIZE = 128
N_HIDDEN = 2048
DROPOUT = 0.5
NB_EPOCH = 10
VERBOSE = 1
OPTIMIZER='sgd'

print('batch_size: ', BATCH_SIZE)
print('n_hidden: ', N_HIDDEN)
print('dropout: ', DROPOUT)
print('nb_epoch: ', NB_EPOCH)
print('verbose: ', VERBOSE)
print ('optimizer: ', OPTIMIZER)

def std_normalize(np_arr):
	return (np_arr - np_arr.mean()) / np_arr.std() 

def min_max_normalize(np_arr):
	return (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())

# load training in a panda dataframe and skip first line
train = pd.read_csv('./data/defaulCC.csv', header=1)
# split X, y
X = train.iloc[:,:-1].values
y = train.iloc[:,-1:].values
dimof_input = X.shape[1]
nb_classes = len(set(y.flat))
print('nb_classes: ', nb_classes)

X_train, X_test, y_train, y_test = \
 train_test_split(X, y, train_size=0.9, random_state=0)

# float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize 
#X_train = std_normalize(X_train)
#X_test = std_normalize(X_test)
X_train = min_max_normalize(X_train)
X_test = min_max_normalize(X_test)

print X_train
print X_test

# convert class vectors to binary class matrices
y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]
print X_train.shape, X_test.shape, y_train.shape, y_test.shape

# this network has dimof_input n the input layer
# N_HIDDEN in the output layer
# dimof_middle in the hidden layer

model = Sequential()
model.add(Dense(N_HIDDEN, input_dim=dimof_input))
model.add(Activation('relu'))	
model.add(Dropout(DROPOUT))

model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))	
model.add(Dropout(DROPOUT))

model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.summary()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, 
	metrics=['accuracy'])

# Train
model.fit(
X_train, y_train,
validation_split=0.2,
batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, verbose=VERBOSE)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=VERBOSE)
print('loss: ', loss)
print('accuracy: ', accuracy)
print()
