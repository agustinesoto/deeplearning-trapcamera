from data_loader.conv_mnist_data_loader import ConvMnistDataLoader
from models.conv_mnist_model import ConvMnistModel
from tuning.vgg16tuning import TuningVGG16
from trainers.simple_mnist_trainer import SimpleMnistModelTrainer


from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
#from tuning.vgg16tuning import TuningVGG16

import tensorflow as tf
from tensorflow import keras


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as e:
        print(e)
        print("missing or invalid arguments")
        exit(0)
    
    print('Create the data generator.')
    data_loader = ConvMnistDataLoader(config) 
    
    print('Getting the models.')
    tuner = TuningVGG16(config)
    tuner.get_models()
    resul = tuner.optimize(data_loader.train_gen,data_loader.val_datagen,data_loader.x_test,data_loader.y_test)
    
if __name__ == '__main__':
    main()
