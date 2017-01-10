import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(1111)  # for reproducibility
 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2, l1, l1l2, activity_l2, activity_l1
from keras.utils.visualize_util import plot
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping

import inspect

#
# save the graph produced by the experiment
#
def print_Graph(
	# Training log
	fitlog, 
	# elapsed time
	elapsed, 
	# input parameters for the experiment
	args, 
	# input values for the experiment
	values):

	experiment_label = "\n".join(['%s=%s' % (i, values[i]) for i in args])
	experiment_file = experiment_label+"-Time= %02d" % elapsed + "sec"
	experiment_file = experiment_file.replace("\n", "-")+'.png'

	fig = plt.figure(figsize=(6, 3))
	plt.plot(fitlog.history["val_acc"])
	plt.title('val_accuracy')
	plt.ylabel('val_accuracy')
	plt.xlabel('iteration')
	fig.text(.7,.15,experiment_label, size='6')
	plt.savefig(experiment_file, format="png")

#
# A LeNet-like convnet for classifying MINST handwritten characters 28x28
#
def convNet_LeNet(

	VERBOSE=1,
	# normlize
	NORMALIZE = True,
	# Network Parameters
	BATCH_SIZE = 128,
	NUM_EPOCHS = 20,
	# Number of convolutional filters 
	NUM_FILTERS = 32,
	# side length of maxpooling square
	NUM_POOL = 2,
	# side length of convolution square
	NUM_CONV = 3,
	# dropout rate for regularization
	DROPOUT_RATE = 0.5,
	# hidden number of neurons first layer
	NUM_HIDDEN = 128,
	# validation data
	VALIDATION_SPLIT=0.2, # 20%
	# optimizer used
	OPTIMIZER = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
	# regularization
	REGULARIZER = l2(0.01)
	): 

	# Output classes, number of MINST DIGITS
	NUM_CLASSES = 10
	# Shape of an MINST digit image
	SHAPE_X, SHAPE_Y = 28, 28
	# Channels on MINST
	IMG_CHANNELS = 1

	# LOAD the MINST DATA split in training and test data
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train = X_train.reshape(X_train.shape[0], 1, SHAPE_X, SHAPE_Y)
	X_test = X_test.reshape(X_test.shape[0], 1, SHAPE_X, SHAPE_Y)

	# convert in float32 representation for GPU computation
	X_train = X_train.astype("float32")
	X_test = X_test.astype("float32")

	if (NORMALIZE):
		# NORMALIZE each pixerl by dividing by max_value=255
		X_train /= 255
		X_test /= 255
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')
	 
	# KERAS needs to represent each output class into OHE representation
	Y_train = np_utils.to_categorical(Y_train, NUM_CLASSES)
	Y_test = np_utils.to_categorical(Y_test, NUM_CLASSES)

	nn = Sequential()
	 
	#FIRST LAYER OF CONVNETS, POOLING, DROPOUT
	#  apply a NUM_CONV x NUM_CONF convolution with NUM_FILTERS output
	#  for the first layer it is also required to define the input shape
	#  activation function is rectified linear 
	nn.add(Convolution2D(NUM_FILTERS, NUM_CONV, NUM_CONV, 
		input_shape=(IMG_CHANNELS, SHAPE_X, SHAPE_Y) ))
	nn.add(Activation('relu'))
	nn.add(Convolution2D(NUM_FILTERS, NUM_CONV, NUM_CONV))
	nn.add(Activation('relu'))
	nn.add(MaxPooling2D(pool_size = (NUM_POOL, NUM_POOL)))
	nn.add(Dropout(DROPOUT_RATE))

	#SECOND LAYER OF CONVNETS, POOLING, DROPOUT 
	#  apply a NUM_CONV x NUM_CONF convolution with NUM_FILTERS output
	nn.add(Convolution2D( NUM_FILTERS, NUM_CONV, NUM_CONV))
	nn.add(Activation('relu'))
	nn.add(Convolution2D(NUM_FILTERS, NUM_CONV, NUM_CONV))
	nn.add(Activation('relu'))
	nn.add(MaxPooling2D(pool_size = (NUM_POOL, NUM_POOL) ))
	nn.add(Dropout(DROPOUT_RATE))
	 
	# FLATTEN the shape for dense connections 
	nn.add(Flatten())
	 
	# FIRST HIDDEN LAYER OF DENSE NETWORK
	nn.add(Dense(NUM_HIDDEN))  
	nn.add(Activation('relu'))
	nn.add(Dropout(DROPOUT_RATE))          

	# OUTFUT LAYER with NUM_CLASSES OUTPUTS
	# ACTIVATION IS SOFTMAX, REGULARIZATION IS L2
	nn.add(Dense(NUM_CLASSES, W_regularizer=REGULARIZER))
	nn.add(Activation('softmax') )

	#summary
	nn.summary()
	#plot the model
	plot(nn)

	# set an early-stopping value
	early_stopping = EarlyStopping(monitor='val_loss', patience=2)

	# COMPILE THE MODEL
	#   loss_function is categorical_crossentropy
	#   optimizer is parametric
	nn.compile(loss='categorical_crossentropy', 
		optimizer=OPTIMIZER, metrics=["accuracy"])

	start = time.time()
	# FIT THE MODEL WITH VALIDATION DATA
	fitlog = nn.fit(X_train, Y_train, \
		batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, \
		verbose=VERBOSE, validation_split=VALIDATION_SPLIT, \
		callbacks=[early_stopping])
	elapsed = time.time() - start

	# Test the network
	results = nn.evaluate(X_test, Y_test, verbose=VERBOSE)
	print('accuracy:', results[1])

	# just to get the list of input parameters and their value
	frame = inspect.currentframe()
	args, _, _, values = inspect.getargvalues(frame)
	# used for printing pretty arguments

	print_Graph(fitlog, elapsed, args, values)

	return fitlog  

# 2 epochs
#log = convNet_LeNet(OPTIMIZER = 'Adam', NUM_EPOCHS=2)
#print(log.history)
# 20 epochs
#log = convNet_LeNet(OPTIMIZER = 'Adam', NUM_EPOCHS=20)
#print(log.history)
# default optimizer = SGD
#log = convNet_LeNet(NUM_EPOCHS=20)
#print(log.history)
# default optimizer = RMSProp
#log = convNet_LeNet(OPTIMIZER=RMSprop(), NUM_EPOCHS=20)
#print(log.history)
## default optimizer 
#log = convNet_LeNet(OPTIMIZER='Adam', DROPOUT_RATE=0)
#print(log.history)
# default optimizer 
#log = convNet_LeNet(OPTIMIZER='Adam', DROPOUT_RATE=0.1)
#print(log.history)
# default optimizer 
#log = convNet_LeNet(OPTIMIZER='Adam', DROPOUT_RATE=0.2)
#print(log.history)
# default optimizer 
#log = convNet_LeNet(OPTIMIZER='Adam', DROPOUT_RATE=0.4)
#print(log.history)
# default optimizer 
#log = convNet_LeNet(OPTIMIZER='Adam', BATCH_SIZE=64)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', BATCH_SIZE=128)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', BATCH_SIZE=256)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', BATCH_SIZE=512)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', BATCH_SIZE=1024)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', BATCH_SIZE=2048)
#print(log.history)
#
#log = convNet_LeNet(OPTIMIZER='Adam', BATCH_SIZE=4096)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', VALIDATION_SPLIT=0.8)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', VALIDATION_SPLIT=0.6)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', VALIDATION_SPLIT=0.4)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', VALIDATION_SPLIT=0.2)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', VALIDATION_SPLIT=0.2, NORMALIZE=False)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', VALIDATION_SPLIT=0.2, NUM_FILTERS=64)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', NUM_FILTERS=128)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', NUM_FILTERS=256)
#print(log.history)
# x log = convNet_LeNet(OPTIMIZER='Adam', NUM_POOL=4)
# x print(log.history)
# x log = convNet_LeNet(OPTIMIZER='Adam', NUM_POOL=8)
# x print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', NUM_CONV=4)
#print(log.history)
# x log = convNet_LeNet(OPTIMIZER='Adam', NUM_CONV=8)
# X print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', NUM_HIDDEN=32)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', NUM_HIDDEN=64)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', NUM_HIDDEN=256)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', NUM_HIDDEN=512)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', NUM_HIDDEN=1024)
#print(log.history)
#log = convNet_LeNet(OPTIMIZER='Adam', REGULARIZER=l1(0.01))
#print(log.history)
regular = l1l2(l1=0.01, l2=0.01)
log = convNet_LeNet(OPTIMIZER='Adam', REGULARIZER=regular)
print(log.history)



	# VERBOSE=1,
	# # normlize
	# NORMALIZE = True,
	# # Network Parameters
	# BATCH_SIZE = 128,
	# NUM_EPOCHS = 100,
	# # Number of convolutional filters 
	# NUM_FILTERS = 32,
	# # side length of maxpooling square
	# NUM_POOL = 2,
	# # side length of convolution square
	# NUM_CONV = 3,
	# # dropout rate for regularization
	# DROPOUT_RATE = 0.5,
	# # hidden number of neurons first layer
	# N_HIDDEN = 128,
	# # validation data
	# VALIDATION_SPLIT=0.2, # 20%
	# # optimizer used
	# OPTIMIZER = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


#plt.show()
