from data_loader.simple_mnist_data_loader import SimpleMnistDataLoader
#from models.simple_mnist_model import SimpleMnistModel
from models.conv_mnist_model import ConvMnistModel

from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args



import tensorflow as tf
from tensorflow import keras


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = ConvMnistDataLoader(config) # ojo

    print('Create the model.')
    model = ConvMnistModel(config)

    print('Create the trainer')
    
    trainer = SimpleMnistModelTrainer(config,model.model, data_loader.get_train_data(),data_loader.get_val_data())
 
    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
