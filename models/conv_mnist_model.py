from comet_ml import Experiment


import tensorflow as tf
from tensorflow import keras
import os
from base.base_model import BaseModel

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard



class ConvMnistModel(BaseModel):
    def __init__(self, config):
        super(ConvMnistModel, self).__init__(config)
        self.config = config
        self.build_model()
        self.init_callbacks()


    def init_callbacks(self):
        self.model.callbacks = []
        
        self.model.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.model.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        

        
        
        

    def build_model(self):
        width_shape = 224
        height_shape = 224
        image_input = Input(shape=(width_shape, height_shape,3))
        conv_base = VGG16(input_tensor=image_input,include_top=False,weights='imagenet')
        
        #First model
        for layer in conv_base.layers:
            layer.trainable = False

        self.model = Sequential()
        self.model.add(conv_base)
        self.model.add(Flatten())

        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dense(1,activation='sigmoid')) 

       
        self.model.summary()



        self.model.compile(
              loss='binary_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['acc'],
        )
        



        
            
            