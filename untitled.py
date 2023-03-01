#!/usr/bin/env python3

import pdb

#importing required libraries
import numpy as np
import tensorflow as tf
import os
import shutil
import splitfolders 
import pandas as pd
import Augmentor
import itertools

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

root_directory = "dataTID"
camera_directory = "2-PN/J15"
width_shape = 224
height_shape = 224
parent_dir = os.path.join(root_directory,camera_directory)
#flag = int(input("¿already processed the directories and splitted data? Type 1: yes. anything else: no."))

pdb.set_trace()

f = open("control.txt", "w")


def preprocessing_data():
        try:
            path = os.path.join(parent_dir, "non_fantasma")
            os.mkdir(path)
        except:None

        # Set the destination directories for "fantasma" and non-"fantasma" directories
        fantasma_dir = os.path.join(parent_dir,"fantasma")
        non_fantasma_dir = os.path.join(parent_dir,"non_fantasma")


        # Iterate over all directories and subdirectories in the parent directory
        for root, dirs,files in os.walk(parent_dir):
            for file in files:
                filePath = os.path.join(root, file)
                if not "fantasma" in filePath: shutil.move(filePath, non_fantasma_dir)
                elif not "non_fantasma" in filePath:shutil.move(filePath, fantasma_dir)

        #deletting junk directories            
        for subdir in os.listdir(parent_dir):
            # Construct the full path to the subdirectory
            subdir_path = os.path.join(parent_dir, subdir)
            # Check if the subdirectory is a directory and its name is different from "fantasma" and "non_fantasma"
            if os.path.isdir(subdir_path) and subdir not in ["fantasma", "non_fantasma"]:
                # Delete the subdirectory and all its contents recursively
                shutil.rmtree(subdir_path)

        #Splitting folders
        input_folder = os.path.join(root_directory,camera_directory)
        # Split with a ratio.
        splitfolders.ratio(input_folder, output=input_folder,ratio=(.7, .2,.1), 
                           group_prefix=None) # default values

        #Data augmentation
        flag1 = int(input("¿wanna apply data augmentation to the train set? Type 1: yes. anything else: no."))
        
        if(flag1 == 1):
            #class_to_apply = input("great, just type the class to apply DA (fantasma or non_fantasma)")
            #p = Augmentor.Pipeline(os.path.join(parent_dir,"train",class_to_apply,output_directory=os.path.join(parent_dir,"train",class_to_apply))

            #p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)
            #p.flip_left_right(probability=0.6)
            #p.flip_top_bottom(probability=0.4)
            #num_of_samples = 129  #poner la resta entre cantidad para balancear 
            #p.sample(num_of_samples)
            None


        #Moving generated new images
        source = os.path.join(parent_dir,"train",class_to_apply,"output")
        destination = os.path.join(parent_dir,class_to_apply,"fantasma")

        allfiles = os.listdir(source)

        # iterate on all files to move them to destination folder
        for f in allfiles:
            src_path = os.path.join(source, f)
            dst_path = os.path.join(destination, f)
            shutil.move(src_path, dst_path)
        #deletting -ipymb checkpoints   
        os.system('!rm -rf `find -type d -name .ipynb_checkpoints`')
                              
                              

def get_models(num_layers: int,
               min_nodes_per_layer: int,
               max_nodes_per_layer: int,
               node_step_size: int,
               hidden_layer_activation: str = 'relu',
               num_nodes_at_output: int = 1,
               output_layer_activation: str = 'sigmoid') -> list:
    
      #Input
    image_input = Input(shape=(width_shape, height_shape, 3))

    #Convolutional base of VGG16 arquitecture trained with imagenet dataset
    conv_base = VGG16(input_tensor=image_input, include_top=False,weights='imagenet')
    conv_base.summary()
    
    
    node_options = list(range(min_nodes_per_layer, max_nodes_per_layer + 1, node_step_size))
    layer_possibilities = [node_options] * num_layers
    layer_node_permutations = list(itertools.product(*layer_possibilities))
    models = []
    for permutation in layer_node_permutations:
        model = tf.keras.Sequential()
        for layer in conv_base.layers:
            layer.trainable = False
        model.add(conv_base)
        model.add(keras.layers.Flatten())

        model_name = ''

        for nodes_at_layer in permutation:
            model.add(tf.keras.layers.Dense(nodes_at_layer, activation='relu'))
            model_name += f'dense{nodes_at_layer}_'

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model._name = model_name[:-1]
        models.append(model)
    return models

                                  
                                  
def optimize(models: list,
             train_datagen: keras.preprocessing.image.DirectoryIterator,
             val_datagen:keras.preprocessing.image.DirectoryIterator,
             X_test: np.ndarray,
             y_test: np.ndarray,
             epochs: int = 10,
             verbose: int = 0) -> pd.DataFrame:
    
    # We'll store the results here
    results = []
    
    def train(model: tf.keras.Sequential) -> dict:
        # Change this however you want
        model.compile(
            loss=tf.keras.losses.binary_crossentropy,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy')
            ]
        )
        # Train the model
        model.fit_generator(
            train_datagen,
            epochs=epochs,
            validation_data=val_datagen,
            verbose=verbose
        )
        
        # Make predictions on the test set
        preds = model.predict(X_test)
        prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(preds)]
        
        # Return evaluation metrics on the test set
        return {
            'model_name': model.name,
            'test_accuracy': accuracy_score(y_test, prediction_classes),
            'test_precision': precision_score(y_test, prediction_classes),
            'test_recall': recall_score(y_test, prediction_classes),
            'test_f1': f1_score(y_test, prediction_classes)
        }
    
    # Train every model and save results
    for model in models:
        try:
            f.write(model.name, end=' ... ')
            res = train(model=model)
            results.append(res)
        except Exception as e:
            f.write(f'{model.name} --> {str(e)}')
        
    return pd.DataFrame(results)                                 
                                  
                                  
                                  
                                

num_classes = 2
epochs = 50
batch_size = 32
    
all_models = get_models(
num_layers=3, 
    min_nodes_per_layer=64, 
    max_nodes_per_layer=256, 
    node_step_size=64)
    
    
   

    
training_dir = os.path.join(parent_dir,"train")
val_dir = os.path.join(parent_dir,"val")
test_dir = os.path.join(parent_dir,"test")
    
idg_flip = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)
    
datagen_train = idg_flip.flow_from_directory(training_dir,
                                    target_size=(width_shape, height_shape),
                                    subset='training',
                                    class_mode='binary',
                                    batch_size = 32,
                                    shuffle=True
                                    #seed=123,
                                    #save_to_dir='data_test/2-PN/L08_aumented'
                                    )
val_generator = idg_flip.flow_from_directory(
    val_dir,
    target_size=(width_shape, height_shape), 
    batch_size = batch_size,
    class_mode='binary', 
    shuffle=False)
                                  
test_datagen = idg_flip.flow_from_directory(test_dir,
                                    target_size=(width_shape, height_shape),
                                    class_mode='binary',
                                    batch_size = 32,
                                    shuffle=True
                                    #seed=123,
                                    #save_to_dir='data_test/2-PN/L08_aumented'
                                    )
 
from tqdm import tqdm
test_datagen.reset()
x_test, y_test = next(test_datagen)
for i in tqdm(range(int(len(test_datagen.labels)/32))): #100 = batch size
      print(i)
      img, label = next(test_datagen)
      x_test = np.append(x_test, img, axis = 0)
      print("label",label)
      y_test = np.append(y_test, label, axis = 0)

print(x_test.shape,y_test.shape)
print(type(x_test),type(y_test))
                                  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
optimization_results = optimize(
    models=all_models,
    train_datagen=datagen_train,
    val_datagen=val_generator,
    X_test=x_test,
    y_test=y_test)

optimization_results.to_csv("optimization_results.csv",index=False)



                                  


