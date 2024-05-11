"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

import hyperparameters as hp


class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, task):

        self.data_path = data_path
        self.task = task

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Mean and std for standardization
        self.mean = np.zeros((224, 224, 3))
        self.std = np.ones((224, 224, 3))
        self.calc_mean_and_std()

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"), True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"), False, False)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".jpg"):
                    file_list.append(os.path.join(root, name))

        # Shuffle filepaths
        random.shuffle(file_list)

        # Take sample of file paths
        file_list = file_list[:hp.preprocess_sample_size]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (hp.preprocess_sample_size, 224, 224, 3))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = img.resize((224, 224))
            img = np.array(img, dtype=np.float32)
            img /= 255.

            # Grayscale -> RGB
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img

        # Calculate the mean image (mean input data point) from the sample
        mean_image = np.mean(data_sample, axis=0)

        # Subtract the mean image from each image in the sample
        centered_data = data_sample - mean_image

        # Calculate the standard deviation image
        std_image = np.std(centered_data, axis=0)

        # Calculate the standard deviation (used for scaling during normalization)
        std_deviation = np.mean(std_image)

        # Store the mean and standard deviation
        self.mean = mean_image
        self.std = std_deviation

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """

      # Standardize the input image using self.mean and self.std
        standardized_img = (img - self.mean) / self.std

        return standardized_img

        # =============================================================

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """
        img = tf.keras.applications.vgg16.preprocess_input(img)

        return img

    # def custom_preprocess_fn(self, img):
    #     """ Custom preprocess function for ImageDataGenerator. """

    #     if self.task == '3':
    #         img = tf.keras.applications.vgg16.preprocess_input(img)
    #     else:
    #         img = img / 255.
    #         img = self.standardize(img)

    #     if random.random() < 0.3:
    #         img = img + tf.random.uniform(
    #             (hp.img_size, hp.img_size, 1),
    #             minval=-0.1,
    #             maxval=0.1)

    #     return img

    def get_data(self, path, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        if augment:
             data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
                width_shift_range=0.1,        # Random horizontal crop by 10% of the width
                height_shift_range=0.1,       # Random vertical crop by 10% of the height
                rotation_range=10,            # Random rotation in the range [-10, 10] degrees
                # zoom_range=0.1,               # Random scaling/zoom between [0.9, 1.1]
                # brightness_range=[0.8, 1.2],  # Random brightness adjustment
                # shear_range=0.1,              # Shear intensity (angle in radians)
                # fill_mode='nearest'           # Fill mode for points outside the input boundaries
            )

            # ============================================================
        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # VGG must take images of size 224x224
        img_size_w, img_size_h = 224, 224

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size_w, img_size_h),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(
                    data_gen.class_indices[img_class])
                self.classes[int(
                    data_gen.class_indices[img_class])] = img_class

        return data_gen
