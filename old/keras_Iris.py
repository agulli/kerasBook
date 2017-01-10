#
# working
#
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

# network and training
N_HIDDEN = 512
NB_EPOCH = 100
BATCH_SIZE = 100
VERBOSE = 1
VALIDATION_SPLIT=0.2

def one_hot_encode_object_array(arr, classes):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, classes)

#load and show iris
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species')
sns.plt.show()

#features, and true label
X = iris.values[:, 0:4]
y = iris.values[:, 4]

#train test
X_train, X_test, y_train, y_test = \
 train_test_split(X, y, train_size=0.7, random_state=0)

#hot encode
nb_classes = len(np.unique(y))
print(nb_classes, 'classes')

y_train_ohe = one_hot_encode_object_array(y_train, nb_classes)
y_test_ohe = one_hot_encode_object_array(y_test, nb_classes)

# 4 input for the features
# 3 output for the classes/species
# 16 hidden layers, sigmoid activation, dense
# a final layer with softmax for picking best one
# optimization based on loss cross-entropy

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(4,)))
model.add(Activation('sigmoid'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', 
    metrics=['accuracy'],
	optimizer='adam')

# fit the model
model.fit(X_train, y_train_ohe, verbose=VERBOSE, nb_epoch=NB_EPOCH,
	batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

# loss, accuracy
loss, accuracy = model.evaluate(X_test, y_test_ohe, verbose=VERBOSE)

print('loss: ', loss)
print('accuracy: ', accuracy)
print()
