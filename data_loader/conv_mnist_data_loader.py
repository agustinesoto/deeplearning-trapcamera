from base.base_data_loader import BaseDataLoader
import os
import shutil
import splitfolders 
import Augmentor
from tqdm import tqdm
import tensorflow as tf
import numpy as np

#root_directory = "dataTID"
#camera_directory = "2-PN/I23"

class ConvMnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ConvMnistDataLoader,self).__init__(config)
        self.root_dir = "dataTID"
        self.camera_dir = "2-PN/J15"
        width_shape = 224
        height_shape = 224

        try:
            directories_preprocesing(self.root_dir,self.camera_dir)
            splitting_images(self.root_dir,self.camera_dir)
            data_augmentation(self.root_dir,self.camera_dir)
        except:None
        
        parent_dir = os.path.join(self.root_dir,self.camera_dir)
    
        
        training_dir = os.path.join(parent_dir,"train")
        val_dir = os.path.join(parent_dir,"val")
        test_dir = os.path.join(parent_dir,"test")
        
        idg_flip = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        self.train_gen = idg_flip.flow_from_directory(training_dir,
                                    target_size=(width_shape, height_shape),
                                    subset='training',
                                    class_mode='binary',
                                    batch_size = 32,
                                    shuffle=True
                                    #seed=123,
                                    #save_to_dir='data_test/2-PN/L08_aumented'
                                    )
        
        self.val_datagen = idg_flip.flow_from_directory(
                            val_dir,
                            target_size=(width_shape, height_shape), 
                            batch_size = 32, #ojo
                            class_mode='binary', 
                            shuffle=False)
        
        
        test_datagen = idg_flip.flow_from_directory(test_dir,
                                    target_size=(width_shape, height_shape),
                                    class_mode='binary',
                                    batch_size = 32,
                                    shuffle=True
                                    #seed=123,
                                    #save_to_dir='data_test/2-PN/L08_aumented'
                                    )
        
        
        test_datagen.reset()
        self.x_test, self.y_test = next(test_datagen)
        for i in tqdm(range(int(len(test_datagen.labels)/32))): #100 = batch size
          img, label = next(test_datagen)
          self.x_test = np.append(self.x_test, img, axis = 0)
          self.y_test = np.append(self.y_test, label, axis = 0)


    def get_train_data(self):
        return self.train_gen
    
    def get_val_data(self):
        return self.val_datagen
        

    def get_test_data(self):
        return self.x_test, self.y_test

    

def directories_preprocesing(root_directory,camera_directory):
    parent_dir = os.path.join(root_directory,camera_directory)
    #Creating non_fantasma directory in camera directory
    try:
        path = os.path.join(parent_dir, "non_fantasma")
        os.mkdir(path)
    except:None
    
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

            
def splitting_images(root_directory,camera_directory):
    input_folder = os.path.join(root_directory,camera_directory)
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    #Train, val, tes
    splitfolders.ratio(input_folder, output=input_folder,ratio=(.7, .2,.1), 
                       group_prefix=None) # default values
    
def data_augmentation(root_directory,camera_directory):
    
    parent_dir = os.path.join(root_directory,camera_directory)
    
    p= Augmentor.Pipeline(os.path.join(parent_dir,"train","fantasma"),output_directory=os.path.join(parent_dir,"train","fantasma"))
    
    p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)
    p.flip_left_right(probability=0.6)
    p.flip_top_bottom(probability=0.4)
    num_of_samples = 129  #poner la resta entre cantidad para balancear 
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
    #deletting -ipymb checkpoints   
    os.system('!rm -rf `find -type d -name .ipynb_checkpoints`')
        

