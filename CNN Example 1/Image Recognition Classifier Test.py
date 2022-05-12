
# Guide from:
#   https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd

# Convolutional Neural Network
# Importing the Keras libraries and packages
from multiprocessing import pool
from pickletools import optimize
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binaru_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'Dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = train_datagen.flow_from_directory(
    'Dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

from IPython.display import display
from PIL import Image

classifier.fit_generator(
    training_set,
    steps_per_epoch=8000,
    epochs=10,
    validation_data=test_set,
    validation_steps=800)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('random.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
if result[0][0] >= 0.5:
    prediction= 'dog'
else:
    prediction = 'cat'
print(prediction)
print("\nEnd of Program !")
