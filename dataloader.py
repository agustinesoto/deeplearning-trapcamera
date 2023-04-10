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
        
    
        
    except Exception as e:
        print(e)
        

if __name__ == '__main__':
    main()
