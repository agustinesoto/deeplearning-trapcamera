from base.base_data_loader import BaseDataLoader
import os
import shutil
import pandas as pd
import Augmentor
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
        data_augmentation(root_dir,camera_dir)
        
        
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
        
    # Split data into training, validation, and test sets
    train_size = 0.7
    valid_test_size = 0.3
    test_size = 0.2
    
    num_samples = len(df)
    train_end_index = int(num_samples * train_size)
    valid_test_start_index = train_end_index + 1
    valid_test_end_index = int(num_samples * (train_size + valid_test_size))

    train_df = df[:train_end_index]
    valid_test_df = df[valid_test_start_index:valid_test_end_index]
    test_df = valid_test_df.sample(frac=test_size/(valid_test_size)) # optional: random_state
    valid_df = valid_test_df.drop(test_df.index)
        
    train_df.to_csv("train.csv",index=False)
    test_df.to_csv("test.csv",index=False)
    valid_df.to_csv("valid.csv",index=False)
    
    
    # Create directories for train, test, and validation data
    # Create directories for train, test, and validation data
    os.makedirs(os.path.join(data_dir, 'splitted_data', 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'splitted_data', 'test'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'splitted_data', 'valid'), exist_ok=True)

    
    print("IVOVVSDJJENFE")
    # Copy images to train directory
    for filename in train_df['filename']:
        label = train_df[train_df['filename']==filename]['label'].values[0]
        src_path = os.path.join(data_dir, label, filename)
        dst_path = os.path.join(data_dir, 'splitted_data', 'train', label, filename) # corregir la ruta de destino
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copyfile(src_path, dst_path)

    # Copy images to test directory
    for filename in test_df['filename']:
        label = test_df[test_df['filename']==filename]['label'].values[0]
        src_path = os.path.join(data_dir, label, filename)
        dst_path = os.path.join(data_dir, 'splitted_data', 'test', label, filename) # corregir la ruta de destino
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copyfile(src_path, dst_path)

    # Copy images to validation directory
    for filename in valid_df['filename']:
        label = valid_df[valid_df['filename']==filename]['label'].values[0]
        src_path = os.path.join(data_dir, label, filename)
        dst_path = os.path.join(data_dir, 'splitted_data', 'valid', label, filename) # corregir la ruta de destino
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copyfile(src_path, dst_path)

        
            
    # Report of count each class 
    train_counts = train_df['label'].value_counts()
    test_counts = test_df['label'].value_counts()
    valid_counts = valid_df['label'].value_counts()

    report_df = pd.DataFrame({
        'train': train_counts,
        'test': test_counts,
        'valid': valid_counts
    })

    # Guardar dataframe como CSV
    report_df.to_csv(os.path.join(root_dir,camera_dir,"images_report.csv"), index=True)



def data_augmentation(root_dir,camera_dir):
    parent_dir = os.path.join(root_dir,camera_dir,"splitted_data")
    
    p= Augmentor.Pipeline(os.path.join(parent_dir,"train","fantasma"))
    
    p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)
    p.flip_left_right(probability=0.6)
    #p.flip_top_bottom(probability=0.4)
    
    results_df = pd.read_csv(os.path.join(root_dir,camera_dir,"images_report.csv"),index_col=0)
        
    num_of_samples = results_df.loc["non_fantasma", "train"] - results_df.loc["fantasma", "train"]
    p.sample(num_of_samples)
    
    #moving generated new images
    source = os.path.join(parent_dir,"train","fantasma","output")
    destination = os.path.join(parent_dir,"train","fantasma")

    import shutil
    # gather all files
    allfiles = os.listdir(source)

    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        shutil.move(src_path, dst_path)
        
    print("listo")
    
    



  
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