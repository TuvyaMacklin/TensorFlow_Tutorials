import tensorflow as tf

import tensorflow_datasets as tfds

import os

from PIL import Image
from tqdm import tqdm

ROOT_DIR = "/home/ec2-user/Documents/datasets"

def process_data_from_class_subset(label_subset, dataset_title):
    '''
    Processes the COCO dataset to only include images that are labeled with one of the labels in the label subset.
    The images are cropped to only include the object they are labeled with.

    The images are saved to a directory named after the dataset title and this is saved in a directory specified by the ROOT_DIR constant. The directory contains a directory for each split. Each split directory contains a directory for each class. Each class directory contains the images that are labeled with that class.

    Params:
        - `label_subset`: a list of labels to include in the dataset
        - `dataset_title`: the name of the dataset to create a directory for
    
    Returns:
        `None`
    
    Side Effects:
        - Creates a directory to hold the dataset and a directory in it to hold each split
        - Saves the cropped images to the appropriate class directory
    
    Precondition:
        The label_subset must be a subset of the labels in the COCO dataset
    
    Example:
        ```
        import process_coco_data

        vehicle_labels = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        process_coco_data.process_data_from_class_subset(vehicles, "vehicles")
        ```

    Jokes from the programmer:
        - Why did the programmer quit his job?
            - Because he didn't get arrays
        - Why did the programmer get stuck in the shower?
            - Because the instructions on the shampoo bottle said "Lather, Rinse, Repeat"
    '''

    ds_train, info = tfds.load("coco", split = "train", shuffle_files = True, with_info= True)
    ds_val, info = tfds.load("coco", split = "validation", shuffle_files = True, with_info= True)
    ds_test, info = tfds.load("coco", split = "test", shuffle_files = True, with_info= True)

    all_labels = info.features["objects"]["label"].names
    subset_ids = _get_label_ids(label_subset, all_labels)

    label_subset_registry = {id: name for id, name in zip(subset_ids, label_subset)}
    ds_dir = os.path.join(ROOT_DIR, dataset_title)

    # Filter out examples that don't have a label in the label subset

    _preprocess_data(ds_train, label_subset_registry, ds_dir, "train")
    _preprocess_data(ds_val, label_subset_registry, ds_dir, "validation")
    _preprocess_data(ds_test, label_subset_registry, ds_dir, "test")

def _preprocess_data(dataset, label_subset_registry, ds_dir, split_name):
    '''
    Preprocesses the dataset by filtering out examples that don't have a label in the label subset, cropping the images to only include the object they are labeled with, and saving them to the appropriate class directory

    Params:
        - `dataset`: the dataset to preprocess
        - `label_subset_registry`: a dictionary mapping label ids to label names
        - `ds_dir`: the directory to save the preprocessed dataset to
        - `split_name`: the name of the split to preprocess
    
    Returns:
        `None`
    
    Side Effects:
        - Creates a directory to hold the split and a directory in it to hold each class
        - Saves the cropped images to the appropriate class directory
    
    Example:
        ```
        _preprocess_data(ds_train, label_subset_registry, ds_dir, "train")
        ```
    
    Notes:
        - This function is not meant to be called directly. It is meant to be called by `process_data_from_class_subset`
    '''

    # Create a directory to hold this split and a directory in it to hold each class
    split_directory = _create_directories(ds_dir, split_name, label_subset_registry.values())

    # Filter out any images that are labeled with something in the subset
    # Crop them to only include the object they are labeled with
    # Save them

    for example in tqdm(dataset, desc = "Preparing the " + split_name + " split"):
        if _within_subset(example, label_subset_registry.keys()):
            _crop_and_save(example, split_directory, label_subset_registry)

def _get_label_ids(user_labels, all_labels):
    '''
    Returns a list of label ids for the labels in the user_labels list

    Params:
        - `user_labels`: a list of labels
        - `all_labels`: a list of all the labels in the COCO dataset
    
    Returns:
        A list of label ids for the labels in the user_labels list
    
    Example:
        ```
        vehicle_labels = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        label_ids = _get_label_ids(vehicle_labels, all_labels)
        ```
    
    Notes:
        - This function is not meant to be called directly. It is meant to be called by `process_data_from_class_subset`
    '''

    id_subset = []

    for user_label in user_labels:
        for index, coco_label in enumerate(all_labels):
            if user_label == coco_label:
                id_subset.append(index)
    
    return id_subset

def _within_subset(example, label_subset_ids):
    '''
    Returns True if the example has a label in the subset

    Params:
        - `example`: an example from the COCO dataset
        - `label_subset_ids`: a list of label ids
    
    Returns:
        True if the example has a label in the subset, False otherwise
    '''
    labels = example["objects"]["label"].numpy()
    
    for label in labels:
        if label in label_subset_ids:
            return True
    
    return False

def _create_directories(ds_dir, split_name, label_subset):
    '''
    Creates a directory to hold the split and a directory in it to hold each class

    Returns the path to the split directory

    Params:
        - `ds_dir`: the directory to save the preprocessed dataset to
        - `split_name`: the name of the split to preprocess
        - `label_subset`: a list of labels
    
    Returns:
        The path to the split directory
    
    Example:
        ```
        split_directory = _create_directories(ds_dir, split_name, label_subset)
        ```
    
    Notes:
        - This function is not meant to be called directly. It is meant to be called by `_preprocess_data`
    '''
    def ensure_directory_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    split_dir = os.path.join(ds_dir, split_name)

    for label in label_subset:
        ensure_directory_exists(os.path.join(split_dir, label))
    
    return split_dir

def _crop_and_save(example, split_directory, label_subset_registry):
    '''
    Crops the image to only include the object it is labeled with and saves it to the appropriate class directory

    Params:
        - `example`: an example from the COCO dataset
        - `split_directory`: the directory to save the preprocessed dataset to
        - `label_subset_registry`: a dictionary mapping label ids to label names

    Returns:
        `None`
    
    Side Effects:
        Saves the cropped image to the appropriate class directory
    
    Example:
        ```
        _crop_and_save(example, split_directory, label_subset_registry)
        ```
    
    Notes:
        - This function is not meant to be called directly. It is meant to be called by `_preprocess_data`
    
    Jokes:
        - Why did the image go to jail?
            - Because it was framed
        
        - Why did the image go to the bar?
            - To get a byte to eat
    '''
    # Find the first label on the example that is in the label subset
    labels = example["objects"]["label"].numpy()
    index = -1
    image_class = None
    for label_index, label_id in enumerate(labels):
        if label_id in label_subset_registry.keys():
            index = label_index
            image_class = label_subset_registry[label_id]
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

    # Tensorflow stores the bounds as [y0, x0, y1, x1]
    # PIL expects the bounds to be [x0, y0, x1, y1]
    bounds[0], bounds[1] = bounds[1], bounds[0]
    bounds[2], bounds[3] = bounds[3], bounds[2]

    # Convert the bounds from a fraction of the image size to a pixel value
    bounds[0] *= width
    bounds[1] *= height
    bounds[2] *= width
    bounds[3] *= height

    # Crop the image
    img = img.crop(bounds)

    # Save the image to the class directory within the split directory
    img.save(path_to_image)