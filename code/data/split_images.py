import os
import shutil
import random

def split_images(input_dir, output_train_dir, output_test_dir, split_ratio=0.9):
    # Create output directories if they don't exist
    if not os.path.exists(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.exists(output_test_dir):
        os.makedirs(output_test_dir)
    
    # Get list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    # Shuffle the list of image files
    random.shuffle(image_files)
    
    # Calculate the number of images for training and testing
    num_images = len(image_files)
    num_train = int(num_images * split_ratio)
    num_test = num_images - num_train
    
    # Allocate images for training and testing
    train_images = image_files[:num_train]
    test_images = image_files[num_train:]
    
    # Copy images to the output directories with sequential renaming
    for idx, img_file in enumerate(train_images, start=1):
        src_path = os.path.join(input_dir, img_file)
        dst_path = os.path.join(output_train_dir, f"{idx}.jpg")
        shutil.copyfile(src_path, dst_path)
    
    for idx, img_file in enumerate(test_images, start=1):
        src_path = os.path.join(input_dir, img_file)
        dst_path = os.path.join(output_test_dir, f"{idx}.jpg")
        shutil.copyfile(src_path, dst_path)

# Example usage:
input_dir_train = "train/none"
input_dir_test = "test/none"
output_train_dir = "train/none"
output_test_dir = "test/none"

split_images(input_dir_train, output_train_dir, output_test_dir)
