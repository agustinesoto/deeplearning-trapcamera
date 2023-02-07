from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
import os
import shutil
import splitfolders 
import Augmentor

root_directory = "dataTID"
camera_directory = "2-PN/I23"

class ConvMnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(ConvMnistDataLoader, self).__init__(config)
        #(self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        #self.X_train = self.X_train.reshape((-1, 28, 28, 1))
        #self.X_test = self.X_test.reshape((-1, 28, 28, 1))
    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

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
            
    for subdir in os.listdir(parent_dir):
        # Construct the full path to the subdirectory
        subdir_path = os.path.join(parent_dir, subdir)
        # Check if the subdirectory is a directory and its name is different from "fantasma" and "non_fantasma"
        if os.path.isdir(subdir_path) and subdir not in ["fantasma", "non_fantasma"]:
            # Delete the subdirectory and all its contents recursively
            shutil.rmtree(subdir_path)

    
def splitting_images(root_directory,camera_directory):
    input_folder = os.path.join(root_directory,camera_directory)
    #Creating non_fantasma directory in camera directory
    try:
        path = os.path.join(input_folder, "splitted_data")
        os.mkdir(path)
    except:None

    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    #Train, val, tes
    splitfolders.ratio(input_folder, output=path,ratio=(.7, .2,.1), 
                       group_prefix=None) # default values
    
def data_augmentation(root_directory,camera_directory):
    
    parent_dir = os.path.join(root_directory,camera_directory)
    p = Augmentor.Pipeline(os.path.join(parent_dir,"splitted_data","train","fantasma"))
    p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)
    p.flip_left_right(probability=0.6)
    p.flip_top_bottom(probability=0.4)
    num_of_samples = 443
    p.sample(num_of_samples)
    
    #moving generated new images
    source = os.path.join(parent_dir,"splitted_data","train","fantasma","output")
    destination = os.path.join(parent_dir,"splitted_data","train","fantasma")

    import shutil
    # gather all files
    allfiles = os.listdir(source)

    # iterate on all files to move them to destination folder
    for f in allfiles:
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, f)
        shutil.move(src_path, dst_path)
        

"""
arreglar directorios junk y .pynb_checkpoints
"""