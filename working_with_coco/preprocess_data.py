import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

import os

from PIL import Image

ds_train, info = tfds.load("coco", split = "train", shuffle_files = True, with_info= True)
ds_val, info = tfds.load("coco", split = "validation", shuffle_files = True, with_info= True)
ds_test, info = tfds.load("coco", split = "test", shuffle_files = True, with_info= True)

labels = info.features["objects"]["label"].names

subset_names = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
subset_ids = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

def get_subset_ids():
    ids = []

    for label in subset_names:
        for index, big_label in enumerate(labels):
            if label == big_label:
                ids.append(index)
    
    return ids

LABEL_SUBSET = {id: name for id, name in zip(subset_ids, subset_names)}
ROOT_DIR = "/home/ec2-user/Documents/datasets/coco/"

# Filter out examples that don't have a label in the label subset

def preprocess_data(dataset, split_name):
    '''
    Preprocesses the dataset by filtering out examples that don't have a label in the label subset
    and cropping the images to only include the object they are labeled with

    Params:
        dataset: the dataset to preprocess
        split_name: the name of the split to create a directory for

    Returns:
        None
    
    Side Effects:
        Creates a directory to hold the split and a directory in it to hold each class
        Saves the cropped images to the appropriate class directory
    '''

    # Create a directory to hold this split and a directory in it to hold each class
    split_directory = create_directories(split_name)

    # Filter out any images that are labeled with something in the subset
    # Crop them to only include the object they are labeled with
    # Save them

    for example in dataset:
        if within_subset(example):
            crop_and_save(example, split_directory)

def within_subset(example):
    '''
    Returns True if the example has a label in the subset
    '''
    labels = example["objects"]["label"].numpy()
    
    for label in labels:
        if label in LABEL_SUBSET.keys():
            return True
    
    return False

def crop_and_save(example, split_directory):
    '''
    Crops the image to only include the object it is labeled with and saves it to the appropriate class directory

    Params:
        example: the example to crop and save
        split_directory: the directory to save the cropped image to

    Returns:
        None
    
    Side Effects:
        Saves the cropped image to the appropriate class directory
    '''
    # Find the first label on the example that is in the label subset
    labels = example["objects"]["label"].numpy()
    index = -1
    image_class = None
    for label_index, label in enumerate(labels):
        if label in LABEL_SUBSET.keys():
            index = label_index
            image_class = LABEL_SUBSET[label]
            break

    # Save the image
    path_to_image = os.path.join(split_directory, image_class, str(example["image/id"].numpy()) + ".jpeg")
    tf.keras.preprocessing.image.save_img(path_to_image, example["image"])

    # Find the corresponding bounding box
    bounds = example["objects"]["bbox"][index].numpy()
    
    # Crop the image using the bounding box
    img = Image.open(path_to_image)
    height = img.height
    width = img.width

    bounds[0], bounds[1] = bounds[1], bounds[0]
    bounds[2], bounds[3] = bounds[3], bounds[2]

    bounds[0] *= width
    bounds[1] *= height
    bounds[2] *= width
    bounds[3] *= height


    img = img.crop(bounds)

    # Save the image to the class directory within the split directory
    img.save(path_to_image)

def create_directories(split_name):
    '''
    Creates a directory to hold the split and a directory in it to hold each class

    Returns the path to the split directory

    Params:
        split_name: the name of the split to create a directory for
    '''
    def ensure_directory_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    ensure_directory_exists(ROOT_DIR + split_name)

    for _, label in LABEL_SUBSET.items():
        ensure_directory_exists(ROOT_DIR + split_name + "/" + label)
    
    return ROOT_DIR + split_name

preprocess_data(ds_train, "train")
preprocess_data(ds_val, "validation")
preprocess_data(ds_test, "test")