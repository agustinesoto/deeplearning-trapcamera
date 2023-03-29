from comet_ml import Experiment
import tensorflow as tf
from tensorflow import keras
#from models.conv_mnist_model import ConvMnistModel
import os

from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys
import pdb



def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        config = process_config(args.config)
        print(type(config))

        # create the experiments dirs
        create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

        print('Create the data generator.')
        data_loader = factory.create("data_loader."+config.data_loader.name)(config)

        print('Create the model.')
        model = factory.create("models."+config.model.name)(config)

        print('Create the trainer')
        trainer = factory.create("trainers."+config.trainer.name)(config,model.model, data_loader.get_train_data(),data_loader.get_val_data())
        print('Start training the model.')
        pdb.set_trace()
        trained_model = trainer.train()
        print("holaaa")
        print("Create the evaluator")
        #'''expects a string that can be imported as with a module.class name'''
        print("config",config)
        print(data_loader.get_test_data()[0])
        print(data_loader.get_test_data()[1])
        print(trained_model)
        
        evaluator = factory.create("testing."+config.evaluator.name)(config,data_loader.get_test_data()[0],data_loader.get_test_data()[1],trained_model)
        
        evaluator.plot_confusion_matrix()
        
        
        #evaluation
        #trainer.test()

    except Exception as e:
        print(e)
        

if __name__ == '__main__':
    main()
