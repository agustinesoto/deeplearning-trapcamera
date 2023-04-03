from base.base_data_loader import BaseDataLoader
import os
import shutil
import pandas as pd
#import Augmentor
#from tqdm import tqdm
import tensorflow as tf
import numpy as np


class DateDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DateDataLoader,self).__init__(config)
        self.config = config
        
        root_dir = config.data_input.parent_dir 
        camera_dir = config.data_input.camera_dir
        
        width_shape = config.data_input.width_shape
        height_shape = config.data_input.height_shape

       
        directories_preprocesing(root_dir,camera_dir)
        date_splitting(root_dir,camera_dir)
        
        
    def get_train_data(self):
        return self.train_gen
    
    def get_val_data(self):
        return self.val_datagen

    def get_test_data(self):
        return (self.x_test, self.y_test)
    


def date_splitting(root_dir,camera_dir):
    # Load image filenames and capture dates into a pandas DataFrame
    data_dir = os.path.join(root_dir,camera_dir)
    
    print(data_dir)
    df = pd.DataFrame(columns=['filename', 'date', 'label'])
    
    exclude_dirs = ['01', '02', '03', '04', '05', '06', '07', '08', '09'] # this's for junk directories 

    for label in ['fantasma', 'non_fantasma']:
        label_dir = os.path.join(data_dir, label)
        filenames = os.listdir(label_dir)
        filenames = [f for f in os.listdir(label_dir) if not (os.path.isdir(os.path.join(label_dir, f)) and f in exclude_dirs)]
        print(filenames)
        dates = [f.split('.')[0].replace('_', ' ') for f in filenames]
        dates = pd.to_datetime(dates, format='%Y %m %d %H %M %S')
        df = df.append(pd.DataFrame({'filename': filenames, 'date': dates, 'label': label}))

    
    df = df.sort_values('date')
    print(df)
    print("hola?")
    df.to_csv("hola.csv",index=False)
    

    
    
        


    





def directories_preprocesing(root_directory,camera_directory):
    parent_dir = os.path.join(root_directory,camera_directory)

    #Creating non_fantasma directory in camera directory if not exist
    if not (os.path.isdir(os.path.join(parent_dir,"non_fantasma"))):
        path = os.path.join(parent_dir, "non_fantasma")
        
        # Set the destination directories for "fantasma" andnon-"fantasma" directories
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

    else:
        print("splitting directories already done!")