from comet_ml import Experiment

from base.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class SimpleMnistModelTrainer(BaseTrain):
    def __init__(self,config, model, train_datagen, val_datagen):
        print("CONFIG HERE, CONFIG HERE")
        print(config)  # add this line
        print(model)
        print(train_datagen)
        print(val_datagen)

        super(SimpleMnistModelTrainer, self).__init__(config,model, train_datagen,val_datagen)
        #self.config = config
        self.model = model
        self.train_datagen = train_datagen
        self.val_datagen = val_datagen
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        #print("aqui")
        #print(type(self.model))
        #print(hasattr(self.model, 'callbacks'))

       
      
    def train(self):
        history = self.model.fit(
            self.train_datagen,
            validation_data = self.val_datagen,
            epochs=20,
            verbose=True,
            batch_size=32,
            callbacks=self.model.callbacks
        )
        
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
        
        return history
