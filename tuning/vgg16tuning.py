
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import itertools
from base.base_tuning import BaseTuning
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import pandas as pd
import numpy as np
from datetime import datetime




class TuningVGG16(BaseTuning):
    def __init__(self, config):
        super(TuningVGG16, self).__init__(config)
        self.models = []
        self.num_layers = config.tuning.num_layers
        #print("num_layers: ",self.num_layers)
        self.min_nodes_per_layer = config.tuning.min_nodes_per_layer
        #print("min_n: ",self.min_nodes_per_layer)

        self.max_nodes_per_layer = config.tuning.max_nodes_per_layer
        #print("max_n: ",self.max_nodes_per_layer)
        self.node_step_size = config.tuning.node_step_size
        #print("step: ",self.node_step_size)
        self.hidden_layer_activation = 'relu'
        self.num_nodes_at_output = 1
        self.output_layer_activation = 'sigmoid'
        
    def get_models(self):
        width_shape = 224
        height_shape = 224
        
        #Input
        image_input = Input(shape=(width_shape, height_shape, 3))

        #Convolutional base of VGG16 arquitecture trained with imagenet dataset
        conv_base = VGG16(input_tensor=image_input, include_top=False,weights='imagenet')
        conv_base.summary()


        node_options = list(range(int(self.min_nodes_per_layer), int(self.max_nodes_per_layer) + 1, int(self.node_step_size)))
        layer_possibilities = [node_options] * self.num_layers
        layer_node_permutations = list(itertools.product(*layer_possibilities))
        
        for permutation in layer_node_permutations:
            model = keras.Sequential()
            for layer in conv_base.layers:
                layer.trainable = False
            model.add(conv_base)
            model.add(keras.layers.Flatten())

            model_name = ''

            for nodes_at_layer in permutation:
                model.add(keras.layers.Dense(nodes_at_layer, activation='relu'))
                model_name += f'dense{nodes_at_layer}_'

            model.add(keras.layers.Dense(1, activation='sigmoid'))
            model._name = model_name[:-1]
            self.models.append(model)
            
    def optimize(self,
             train_datagen: keras.preprocessing.image.DirectoryIterator,
             val_datagen:keras.preprocessing.image.DirectoryIterator,
             X_test: np.ndarray,
             y_test: np.ndarray,
             epochs: int = 10,
             verbose: int = 0) -> pd.DataFrame:
    
        # We'll store the results here
        results = []
    
        def train(model: keras.Sequential) -> dict:
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
        for model in self.models:
            try:
                print(model.name, end=' ... ')
                res = train(model=model)
                results.append(res)
            except Exception as e:
                print(f'{model.name} --> {str(e)}')
        
        res = pd.DataFrame(results)
        #########
        #now = datetime.now()
        #dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        #########
        res.to_csv(f"optimization_results.csv")
        return pd.DataFrame(results)