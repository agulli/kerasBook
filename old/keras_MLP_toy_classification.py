#
# Multi-layer perceptron (MLP) toy example for 
# classifying a simple four class dataset
#
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.utils.visualize_util import plot

from sklearn import datasets
from sklearn.cross_validation import train_test_split

# dataset
N_SAMPLES = 10000
N_FEATURES = 2
N_CLASSES = 4
#
# network and training
N_HIDDEN = 10
NB_EPOCH = 10
BATCH_SIZE = 100
VERBOSE = 1
VALIDATION_SPLIT=0.2

#generate a syntetic dataset with (N_SAMPLES x n_features) and output N_CLASSES
X, y = datasets.make_classification(
    n_samples=N_SAMPLES, n_features=N_FEATURES, n_redundant=0,
    n_classes=N_CLASSES, n_clusters_per_class=1)

# plot with label userd as color
#plt.scatter(X[:,0], X[:,1], c=y)
#plt.show()

# number of classes contained in Y
nb_classes = np.max(y)+1
print(nb_classes, 'classes')

# create train and test sets
X_train, X_test, y_train, y_test = \
  train_test_split(X, y, train_size=0.9, random_state=0)

# convert to float32, supported by GPUs
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# convert class vectors to binary class matrices:
#    class vector (integers from 0 to nb_classes)
#    to binary class matrix, for use with categorical_crossentrop
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X.shape[1], 'features')
print(nb_classes, 'classes')

# lets define the network
model = Sequential()
# INPUT LAYER: input the features, and output the dimension of hidden layer
model.add(Dense(N_HIDDEN, input_shape=(X_train.shape[1],)))
model.add(Activation('tanh'))
# OUTPUT LAYER: applies a 'softmax' non linearities on N_FEATURE
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# summary
model.summary()

# output the model into a graph
plot(model)

# compile
#   loss function is categorical_crossentropy
#   metric is accuracy
#   optimizer is SGD
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='SGD')

# fit the model
model.fit(X_train, y_train, 
	validation_split=VALIDATION_SPLIT,
	verbose=VERBOSE, 
	nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=VERBOSE)
print('loss: ', loss)
print('accuracy: ', accuracy)
print()
