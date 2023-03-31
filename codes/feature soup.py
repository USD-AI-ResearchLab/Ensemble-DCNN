import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
import cv2 as cv
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras import Model
from keras.utils.vis_utils import plot_model
import pathlib
import shutil

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, auc, classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
import seaborn as sns

"""Variables"""

train_dir = 'Train_Images'
val_dir = 'Val_Images'
test_dir = 'Test_Images'

#number of classes
num_classes = 2
img_height = 224
img_width = 224
base_ephoc = 50
base_ephoc_executed = base_ephoc
finetune_ephoc = 200
 

base_learning_rate = 0.0001
finetune_learning_rate = 0.0000001

common_input = keras.Input(shape=(224,224,3), name='Common_Input')

vgg19_b = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_tensor=common_input)
resnet_b = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_tensor=common_input)
densenet_b = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False, input_tensor=common_input)
mobile_b = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_tensor=common_input)
inception_b = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=common_input)

#freeze all the baselayers
vgg19_b.trainable = False
resnet_b.trainable = False
mobile_b.trainable = False
inception_b.trainable = False
densenet_b.trainable = False

"""Add the GAP layers to each base mod"""

vgg19_out = tf.keras.layers.GlobalAveragePooling2D()(vgg19_b.output)
resnet152_out = tf.keras.layers.GlobalAveragePooling2D()(resnet_b.output)
mobilenet_out = tf.keras.layers.GlobalAveragePooling2D()(mobile_b.output)
inception_out = tf.keras.layers.GlobalAveragePooling2D()(inception_b.output)
densenet_out = tf.keras.layers.GlobalAveragePooling2D()(densenet_b.output)

"""Rename layers to avoid name duplication"""

for layer in vgg19_b.layers:
    layer._name = layer.name + str("_vgg")

for layer in resnet_b.layers:
    layer._name = layer.name + str("_res")

for layer in mobile_b.layers:
    layer._name = layer.name + str("_mob")

for layer in inception_b.layers:
    layer._name = layer.name + str("_inc")

for layer in densenet_b.layers:
    layer._name = layer.name + str("_den")

"""Merge the output of each of the layers"""

mergedOutput = tf.keras.layers.Concatenate()([vgg19_out, resnet152_out, mobilenet_out, inception_out, densenet_out])

added_layer = tf.keras.layers.Dropout(0.4, noise_shape=None, seed=None)(mergedOutput)
added_layer = tf.keras.layers.Dense(1024, activation='relu')(added_layer)
added_layer = tf.keras.layers.Dropout(0.4, noise_shape=None, seed=None)(added_layer)
added_layer = tf.keras.layers.Dense(1024, activation='relu')(added_layer)
added_layer = tf.keras.layers.Dropout(0.4, noise_shape=None, seed=None)(added_layer)
pred_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(added_layer)


#define model
model = Model(inputs=common_input, outputs=pred_layer)

#compile the model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=base_learning_rate),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics = ['accuracy'])

"""Plot Model and Summary"""

#keras.utils.plot_model(model, "concathed.png", show_shapes=True)  
#model.summary()

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_ds = train_datagen.flow_from_directory(train_dir, 
                                            seed = 123,
                                            target_size=(img_height,img_width),
                                            batch_size=32,
                                            class_mode='sparse',
                                            classes=['Intact','Fractured']                   
                                        )


val_ds = train_datagen.flow_from_directory(val_dir, 
                                            seed = 123,
                                            target_size=(img_height,img_width),
                                            batch_size=32,
                                            class_mode='sparse',
                                            classes=['Intact','Fractured']
                                        )
test_ds = train_datagen.flow_from_directory(test_dir, 
                                            seed = 123,
                                            target_size=(img_height,img_width),
                                            batch_size=32,
                                            class_mode='sparse',
                                            classes=['Intact','Fractured']
                                        )

class_indices = train_ds.class_indices

"""#### Callbacks

Checkpoint
"""

#define the model checkpoint path
model_checkpoint_path = os.path.join('Models', 'concat' + '_checkpoint/')

#check for the folder path, if exists skip else create the path
if not (os.path.exists(model_checkpoint_path)):
    os.makedirs(model_checkpoint_path)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

"""Stopping Point"""

stopping_point = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=15,
    verbose=1,
    mode="max",
    restore_best_weights=True,
    start_from_epoch=1
)

"""LR Scheduler"""

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                              patience=5, min_lr=finetune_learning_rate)

"""TB Logs"""

tensorboard_logs = tf.keras.callbacks.TensorBoard(log_dir="./tb_logs")

"""Data Logger"""

#check for the file for saving heatmap
csv_save_path = 'Graphs/' + 'concat'
csv_save_file = csv_save_path + '/' + 'concat' + '_' +'_train_data.csv'
csv_save_file2 = csv_save_path + '/' + 'concat' + '_' +'_fine_data.csv'

if not (os.path.exists(csv_save_path)):
    os.makedirs(csv_save_path)

data_logger = tf.keras.callbacks.CSVLogger(csv_save_file, 
                                            separator= ',',
                                            append=True
                                            )
data_logger2 = tf.keras.callbacks.CSVLogger(csv_save_file2, 
                                            separator= ',',
                                            append=True
                                            )

"""Train"""

try:
    history_base = model.fit(train_ds, validation_data=val_ds, 
                             epochs=base_ephoc, 
                             callbacks=[model_checkpoint_callback, stopping_point, data_logger, reduce_lr, tensorboard_logs])
    base_ephoc_executed = len(history_base.history['loss'])
except ValueError as e:
    print(f"Error: {e}")
    exit()
except RuntimeError as e:
    print(f"Error: {e}")
    exit()

"""Plots"""

def plot_accuracy(acc,val_acc,type):
    #plot for the accuracy
    plt.figure(figsize=(16,8))
    plt.plot(acc,label='Training Accuracy')
    plt.plot(val_acc,label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.plot([base_ephoc_executed-1, base_ephoc_executed-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    #plt.show()

    #define the model checkpoint path
    figure_save_path = 'Graphs/' + 'concat'
    figure_save_name =  figure_save_path + '/' + 'concat' + '_' + type +'_accuracy.jpg'

    #check for the folder path, if exists skip else create the path
    if not(os.path.exists(figure_save_path)):
        os.makedirs(figure_save_path)

    plt.savefig(figure_save_name, dpi=200)

def plot_loss(loss,val_loss,type):
    #plotting the data from ephoc to
    plt.figure(figsize=(16,8))
    plt.plot(loss,label='Training Loss')
    plt.plot(val_loss,label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.plot([base_ephoc_executed-1, base_ephoc_executed-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend()
    plt.title('Training and Validation Loss')
    #plt.show()

    #define the model checkpoint path
    figure_save_path = 'Graphs/' + 'concat'
    figure_save_name =  figure_save_path + '/' + 'concat' + '_' + type +'_loss.jpg'

    #check for the folder path, if exists skip else create the path
    if not(os.path.exists(figure_save_path)):
        os.makedirs(figure_save_path)

    plt.savefig(figure_save_name, dpi=200)

"""Plot Call"""

#add the base acc and loss with the fintuned loss and acc to get the whole of the acc and loss
acc = history_base.history['accuracy']
val_acc = history_base.history['val_accuracy']

loss = history_base.history['loss']
val_loss = history_base.history['val_loss']

plot_accuracy(acc,val_acc,'base')
plot_loss(loss,val_loss,'base')

"""## Fine Tuning the Model Performance

Unfreeze the layers of the basemodel and recompile the model
"""

for layer in vgg19_b.layers[-10:-1]:
    if not isinstance(layer, layers.BatchNormalization):
        print(layer.name)
        layer.trainable = True

for layer in resnet_b.layers[-275:-1]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

for layer in mobile_b.layers[-81:-1]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

for layer in inception_b.layers[-150:-1]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

for layer in densenet_b.layers[-200:-1]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=finetune_learning_rate),
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics = ['accuracy'])
#model.summary()

"""Retrain and Continue Training"""

total_ephocs = base_ephoc_executed + finetune_ephoc
try:
    history_fine_tune = model.fit(train_ds, validation_data=val_ds, 
                             epochs=total_ephocs, 
                             initial_epoch = history_base.epoch[-1],
                             callbacks=[model_checkpoint_callback, stopping_point, data_logger2, reduce_lr, tensorboard_logs])
except ValueError as e:
    print(f"Error: {e}")
    exit()
except RuntimeError as e:
    print(f"Error: {e}")
    exit()

"""Plot the fine tune graph"""

acc += history_fine_tune.history['accuracy']
val_acc += history_fine_tune.history['val_accuracy']

loss += history_fine_tune.history['loss']
val_loss += history_fine_tune.history['val_loss']


#plotting the data from ephoc to
plot_accuracy(acc,val_acc,'finetune')
plot_loss(loss,val_loss,'finetune')

"""Save Model"""

#define the model checkpoint path
model_save_path = os.path.join('Models/' + 'concat')

#check for the folder path, if exists skip else create the path
if not (os.path.exists(model_save_path)):
    os.makedirs(model_save_path)

#create a path for model name
model_save_file = model_save_path + '/' + 'concat_model' + '.h5'

if os.path.exists(model_save_file):
    pass

model.save(model_save_file)

"""Load"""

model = keras.models.load_model(model_save_file)

"""Preds"""

true_classes = test_ds.classes
class_indices = train_ds.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

#pred
prediction = model.predict(test_ds)
pred_classes = np.argmax(prediction, axis=1)

"""Metrices Calculation"""

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, auc, classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
import seaborn as sns

#calculating each of the metrices
accuracy = accuracy_score(true_classes, pred_classes)
balanced_accuracy = balanced_accuracy_score(true_classes, pred_classes)
precision = precision_score(true_classes, pred_classes)
recall = recall_score(true_classes, pred_classes)
f1 = f1_score(true_classes, pred_classes)
auc_mod = roc_auc_score(true_classes, pred_classes)

"""Confusion Matrix"""

conf_mat = confusion_matrix(true_classes, pred_classes)
fig, ax = plt.subplots(figsize=(6,6))
ax = sns.heatmap(conf_mat, annot=True, cbar=False, square=True, fmt='d', cmap=plt.cm.Blues, xticklabels=class_subset, yticklabels=class_subset,annot_kws={'size': 18})
heatmap = ax.get_figure()

#check for the file for saving heatmap
figure_save_path = 'Graphs/' + 'concat' + '/'
figure_save_name = figure_save_path + 'concat' + '_' +'fined_tuned_confusion_matrix.jpg'

#check for the folder path, if exists skip else create the path
if not (os.path.exists(figure_save_path)):
    os.makedirs(figure_save_path)

heatmap.savefig(figure_save_name, dpi=200)

"""Print"""

print("Model Accuracy For Given Model on Test Datset: {:.2f}%".format(accuracy * 100))
print("Balanced Accuracy For Given Model on Test Datset: {:.2f}%".format(balanced_accuracy * 100))
print("Precision For Given Model on Test Datset: {:.2f}%".format(precision * 100))
print("Recall For Given Model on Test Datset: {:.2f}%".format(recall * 100))
print("F1-Score For Given Model on Test Datset: {:.2f}%".format(f1 * 100))


#dictionary of the metrices
metrices = {'Accuracy' : [accuracy], 'Balanced_Accuracy': [balanced_accuracy], 'Precision': [precision], 'Recall': [recall], 'F1_Score': [f1], 'AUC': [auc_mod]}

#create a dataframe of the metrices
df = pd.DataFrame.from_dict(metrices, orient='columns', dtype=None, columns=None)

#check for the file for saving metrices
file_save_path = 'Graphs/' + 'concat' + '/'
file_save_name = file_save_path + 'concat' + '_' +'metrices.csv'

#check for the folder path, if exists skip else create the path
if not (os.path.exists(file_save_path)):
    os.makedirs(file_save_path)

df.to_csv(file_save_name, index=None)