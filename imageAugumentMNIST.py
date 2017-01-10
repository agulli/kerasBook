from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.datasets import mnist

import numpy as np

NUM_TO_AUGMENT=5

#load dataset
# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# instantiate ImageDataGenerator to create approximately 10 images for
# each input training image
print("Augmenting training set images...")
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

xtas, ytas = [], []
for i in range(X_train.shape[0]):
    num_aug = 0

    x = X_train[i] # (28, 28)
    x = x.reshape((1,) + x.shape)  # (1, 32, 32)
    x = x.reshape((1,) + x.shape)  # (1, 1, 32, 32)
 
    for x_aug in datagen.flow(x, batch_size=1,
    	save_to_dir='preview', save_prefix='mnist', save_format='jpeg'):
       if num_aug >= NUM_TO_AUGMENT:
           break
       xtas.append(x_aug[0])
       num_aug += 1

