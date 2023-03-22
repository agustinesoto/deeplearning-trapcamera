from comet_ml import Experiment
import matplotlib.pyplot as plt

from base.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime


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
        def plotTraining(hist, epochs, typeData):
            if typeData=="loss":
                plt.figure(1,figsize=(10,5))
                yc=hist.history['loss']
                xc=range(epochs)
                plt.ylabel('Loss', fontsize=24)
                plt.plot(xc,yc,'-r',label='Loss Training')
            if typeData=="accuracy":
                plt.figure(2,figsize=(10,5))
                yc=hist.history['accuracy']
                for i in range(0, len(yc)):
                    yc[i]=100*yc[i]
                xc=range(epochs)
                plt.ylabel('Accuracy (%)', fontsize=24)
                plt.plot(xc,yc,'-r',label='Accuracy Training')
            if typeData=="val_loss":
                plt.figure(1,figsize=(10,5))
                yc=hist.history['val_loss']
                xc=range(epochs)
                plt.ylabel('Loss', fontsize=24)
                plt.plot(xc,yc,'--b',label='Loss Validate')
            if typeData=="val_acc":
                plt.figure(2,figsize=(10,5))
                yc=hist.history['val_acc']
                for i in range(0, len(yc)):
                    yc[i]=100*yc[i]
                xc=range(epochs)
                plt.ylabel('Accuracy (%)', fontsize=24)
                plt.plot(xc,yc,'--b',label='Training Validate')


            plt.rc('xtick',labelsize=24)
            plt.rc('ytick',labelsize=24)
            plt.rc('legend', fontsize=18) 
            plt.legend()
            plt.xlabel('Number of Epochs',fontsize=24)
            plt.grid(True)

            
            # Create folder with the current date
            date_format = datetime.now().strftime('%Y-%m-%d %H:%M')
            folder_name = os.path.join("experiments_img",date_format)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                
            plt.savefig(f"{folder_name}/{typeData}.png")

        
        
        history = self.model.fit(
            self.train_datagen,
            validation_data = self.val_datagen,
            epochs=100,
            verbose=True,
            batch_size=32
            #callbacks=self.model.callbacks
        )
        
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
        
        plotTraining(history,100,"loss")
        plotTraining(history,100,"acurracy")
        plotTraining(history,100,"val_loss")
        plotTraining(history,100,"val_acc")
        
        return self.model
        
    

    
