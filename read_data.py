import os
import random
import shutil
random.seed(42)

# path to downloaded EuroSAT dataset and path to split location

path_images = '/home/david/Uni/Thesis/EuroSat/Data/2750'
path_split_images = '/home/david/Uni/Thesis/EuroSat/Data/split'

# getting train and validation directories

path_to_train = os.path.join(path_split_images, "train")
path_to_valid = os.path.join(path_split_images, "validation")

# making sub-directories for each class folder

sub_dirs = [sub_dir for sub_dir in os.listdir(path_images)
            if os.path.isdir(os.path.join(path_images, sub_dir))]

validation_split = 0.3

for sub_dir in sub_dirs:
    current_dir = os.path.join(path_images, sub_dir)
    files = os.listdir(current_dir)
    random.shuffle(files)
    split_point = int(len(files) * validation_split)
    validation_files = files[:split_point]
    train_files = files[split_point:]

    # copy files to path_split_images

    if not os.path.isdir(os.path.join(path_to_train, sub_dir)):
        os.makedirs(os.path.join(path_to_train, sub_dir))
    if not os.path.isdir(os.path.join(path_to_valid, sub_dir)):
        os.makedirs(os.path.join(path_to_valid, sub_dir))

    for file in train_files:
        shutil.copy2(os.path.join(current_dir, file),
                     os.path.join(path_to_train, sub_dir))
    for file in validation_files:
        shutil.copy2(os.path.join(current_dir, file),
                     os.path.join(path_to_valid, sub_dir))



