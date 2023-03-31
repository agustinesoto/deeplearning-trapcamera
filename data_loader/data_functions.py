import os
import shutil

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