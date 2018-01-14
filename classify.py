# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 09:46:58 2018

IMPORTANT :
    1. This project is for a specific type of folder arrangement of dataset
    2. Please intall : {1.Theano  ,  2.Tensorflow  ,  3.Keras}  before executing it.
    3. Replace the folder names according to your own requirements
    
NOTE : Dataset is empty but structured ; use your own
@author: Udit
"""
# Convolutional Neural Network


# Importing the required libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


"""
Building a CNN requires the following steps :
    1. Convolution
    2. Pooling
    3. Flattening
    4. Full connection
    5. Compiling
"""

classifier = Sequential() #CNN initialised

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) #Convolution

classifier.add(MaxPooling2D(pool_size = (2, 2))) #Max_Pooling

# Adding a 2nd convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten()) #Flattening

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid')) #Full_connection

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #compiling


"""
Fitting a CNN to our Classifier requires the following steps :
    1. Generate more image data from existing dataset by various transformations
    2. Create batch of training and test test
    3. Fit the classifier
"""

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1./255) #1

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary') #2

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000) #3
