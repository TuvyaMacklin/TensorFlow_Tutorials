'''
A collection of functions for setting up tensorflow models.
'''

import os
from tensorflow import keras
from keras import layers

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

def get_basic_data_augmentation(flip: str = "horizontal", input_shape = None, extra_layers = None):
    '''
    Returns a keras.Sequential object containing basic data augmentation layers.

    Parameters:
        flip (str): The type of flip to perform. Can be "horizontal", "vertical", or "horizontal_and_vertical".
        input_shape (tuple): The shape of the input images.
        extra_layers (list): A list of extra layers to add to the Sequential object.
    
    Returns:
        data_augmentation_layers (keras.Sequential): A keras.Sequential object containing the data augmentation layers.
    
    Example:
        >>> data_augmentation_layers = mutils.get_basic_data_augmentation()
        >>>
        >>> model = tf.keras.Sequential([
        ...     data_augmentation_layers,
        ...     tf.keras.layers.Rescaling(1./255),
        ...     tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        ...     tf.keras.layers.MaxPooling2D(),
                ...
        ...     tf.keras.layers.Flatten(),
        ...     tf.keras.layers.Dense(128, activation='relu'),
        ...     tf.keras.layers.Dense(num_classes)
        ... ])
    '''
    if not input_shape is None:
        flip_layer = layers.RandomFlip(flip, input_shape = input_shape)
    else:
        flip_layer = layers.RandomFlip(flip)

    data_augmentation_layers = keras.Sequential([
        flip_layer,
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    if not extra_layers is None:
        for layer in extra_layers:
            data_augmentation_layers.add(layer)

    return data_augmentation_layers