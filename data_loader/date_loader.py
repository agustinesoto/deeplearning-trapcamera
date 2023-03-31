from base.base_data_loader import BaseDataLoader
import os
import shutil
import Augmentor
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import pdb 


class DateDataLoader(BaseDataLoader):
    
    def __init__(self, config):
        pdb.set_trace()
        super(DateDataLoader,self).__init__(config)
        self.config = config
        
        root_dir = config.data_input.parent_dir 
        camera_dir = config.data_input.camera_dir
        
        width_shape = config.data_input.width_shape
        height_shape = config.data_input.height_shape

       
        directories_preprocesing(root_dir,camera_dir)
        
        
    def get_train_data(self):
        return self.train_gen
    
    def get_val_data(self):
        return self.val_datagen

    def get_test_data(self):
        return (self.x_test, self.y_test)

    