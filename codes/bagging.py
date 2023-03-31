#import for garbage collections
import gc

import numpy as np
import pandas as pd
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
import cv2 as cv
import os
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import Model
from keras.utils.vis_utils import plot_model
import pathlib
import shutil

#base model import
from tensorflow.keras.applications import VGG19 as base_mod

import warnings
warnings.filterwarnings("ignore")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

batch_size = 32
img_height = 224
img_width = 224
num_classes = 2
n_split = 5 #number of folds
test_dir = 'Test_Images'
train_dir = 'Train_Images2'


base_learning_rate = 0.0001
finetune_learning_rate = 0.0000001
base_ephoc = 50
base_ephoc_executed = base_ephoc
total_ephoc = 200

#no of bagged models
no_of_models = 5

"""Load the dataframe"""

df = pd.read_csv('training_images.csv')

"""Stopping point for the model"""

stopping_point = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    verbose=1,
    mode="max",
    restore_best_weights=True,
    start_from_epoch=1
)

"""Learning Rate Scheduler"""

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                              patience=5, min_lr=finetune_learning_rate)

"""Data Logger for each fold"""

#no data logger defined

"""Define the bagged models"""

def create_model(model_type):
    if model_type == 'vgg':
        from tensorflow.keras.applications import VGG19 as base_mod
        base_model = base_mod(input_shape=(img_height,img_width,3), include_top=False, weights='imagenet')
        base_model.trainable = False
        last_base_layer = base_model.get_layer('block5_pool')
        last_layer_output = last_base_layer.output

    elif model_type == 'resnet152':
        from tensorflow.keras.applications import ResNet152V2 as base_mod
        base_model = base_mod(input_shape=(img_height,img_width,3), include_top=False, weights='imagenet')
        base_model.trainable = False
        last_base_layer = base_model.get_layer('post_relu')
        last_layer_output = last_base_layer.output

    elif model_type == 'mobilenet':
        from tensorflow.keras.applications import MobileNetV2 as base_mod
        base_model = base_mod(input_shape=(img_height,img_width,3), include_top=False, weights='imagenet')
        base_model.trainable = False
        last_base_layer = base_model.get_layer('out_relu')
        last_layer_output = last_base_layer.output

    elif model_type == 'inception':
        from tensorflow.keras.applications import InceptionV3 as base_mod
        base_model = base_mod(input_shape=(img_height,img_width,3), include_top=False, weights='imagenet')
        base_model.trainable = False
        last_base_layer = base_model.get_layer('mixed10')
        last_layer_output = last_base_layer.output
    
    else:
        from tensorflow.keras.applications import DenseNet169 as base_mod
        base_model = base_mod(input_shape=(img_height,img_width,3), include_top=False, weights='imagenet')
        base_model.trainable = False
        last_base_layer = base_model.get_layer('relu')
        last_layer_output = last_base_layer.output
    
    added_layer = tf.keras.layers.GlobalAveragePooling2D()(last_layer_output)
    added_layer = tf.keras.layers.Dropout(0.4, noise_shape=None, seed=None)(added_layer)
    added_layer = tf.keras.layers.Dense(1024, activation='relu')(added_layer)
    added_layer = tf.keras.layers.Dropout(0.4, noise_shape=None, seed=None)(added_layer)
    added_layer = tf.keras.layers.Dense(1024, activation='relu')(added_layer)
    added_layer = tf.keras.layers.Dropout(0.4, noise_shape=None, seed=None)(added_layer)
    added_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(added_layer)

    #define the model
    model = tf.keras.Model(base_model.input, added_layer)

    #compile the model
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=base_learning_rate),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics = ['accuracy'])
    #return the mod
    return model

"""Data Generator"""

train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255
                                                                )

test_ds = train_data_generator.flow_from_directory('Test_Images', 
                                            seed = 123,
                                            target_size=(img_height,img_width),
                                            batch_size=32,
                                            class_mode='sparse',
                                            classes=['Intact','Fractured']
                                            )

"""For Recompiling"""

def re_compile(model):
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=finetune_learning_rate),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy'])
    return model

"""Bagging for Each Model"""

from sklearn.model_selection import train_test_split
for i in range(no_of_models):
    split, _ = train_test_split(df, test_size = 0.2) #only take 80% of the whole sample for each model
    train, validate = train_test_split(split, test_size=0.1875) #break the smaple to get the 15% validation split of the original sample
    if i == 0:
        model_type = 'vgg'
    elif i == 1:
        model_type = 'resnet152'
    elif i == 2:
        model_type = 'mobilenet'
    elif i == 3:
        model_type = 'inception'
    else:
        model_type = 'densenet'
    model = create_model(model_type)

    train_set = train_data_generator.flow_from_dataframe(dataframe=train,
                                                x_col="filenames", y_col="labels",
                                                seed = 123,
                                                target_size=(img_height,img_width),
                                                batch_size=batch_size,
                                                class_mode='sparse',
                                                classes=['Intact','Fractured']
                                            )
    validation_set = train_data_generator.flow_from_dataframe(dataframe=validate,
                                                x_col="filenames", y_col="labels",
                                                seed = 123,
                                                target_size=(img_height,img_width),
                                                batch_size=batch_size,
                                                class_mode='sparse',
                                                classes=['Intact','Fractured']
                                            )
    history_base = model.fit(train_set, validation_data=validation_set, epochs=base_ephoc, callbacks=[stopping_point, reduce_lr])

    #get the no of ephocs executed
    base_ephoc_executed = len(history_base.history['loss'])

    #based on the model unfreeze some layers and contiue training
    if model_type == 'vgg':
        for layer in model.layers[-10:-1]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        model = re_compile(model)
        flag = True

    elif model_type == 'resnet152':
        for layer in model.layers[-275:-1]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        model = re_compile(model)
        flag = True
    
    elif model_type == 'mobilenet':
        for layer in model.layers[-81:-1]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        model = re_compile(model)
        flag = True

    elif model_type == 'inception':
        for layer in model.layers[-150:-1]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        model = re_compile(model)
        flag = True
    
    elif model_type == 'densenet':
        for layer in model.layers[-200:-1]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        model = re_compile(model)
        flag = True
    
    #continue if any of the condition met
    if flag==True:
        history_base = model.fit(train_set, validation_data=validation_set, initial_epoch=base_ephoc_executed, epochs=total_ephoc, callbacks=[stopping_point, reduce_lr])
    else:
        pass    
    
    #save each model after each run
    model_save_path = os.path.join('Models/' + model_type)
    #check for the folder path, if exists skip else create the path
    if not (os.path.exists(model_save_path)):
        os.makedirs(model_save_path)

    #create a path for model name
    model_save_file = model_save_path + '/' + model_type + '_' + 'bagged' +'.h5'
    model.save(model_save_file)

    gc.collect()