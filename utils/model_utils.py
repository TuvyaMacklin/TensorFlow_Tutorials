'''
A collection of functions for setting up tensorflow models.
'''

import os
from tensorflow import keras

def get_split_of_dataset_from_directory(data_dir, split, **kwargs):
    '''
    Returns a tf.data.Dataset object from a directory of images.

    Parameters:
        data_dir (str): The path to the directory containing the images.
        split (str): The name of the subdirectory containing the images.
        **kwargs: Keyword arguments to pass to keras.utils.image_dataset_from_directory.
    
    Returns:
        ds (tf.data.Dataset): A tf.data.Dataset object containing the images.
    '''
    ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, split),
        **kwargs
    )
    return ds