# TensorFlow Tutorials
This is a collection of python scripts that were written while learning to use TensorFlow for computer vision problems.

## Utils
Of relevance to people going through the same tutorials is the collection of utility functions I wrote up. These provide many helpful functions for plotting examples, plotting model histories, saving and loading models and more.

## Plot Utils
Functions:
- `plot_accuracy(history)`: Creates a plot of the accuracy of a model with respect to the epochs. 
- `plot_history(history)`: Creates a plot of the history of a model with respect to the epochs. 
- `plot_images(rows, cols, images, label)`: Plots a grid of images. 
- `plot_examples_from_dataset(dataset, num_of_examples)`: Plots examples from a dataset. 

## Model Utils
Functions:
- `get_split_of_dataset_from_directory(data_dir, split, **kwargs)`: Returns a tf.data.Dataset object from a directory of images.
- `get_basic_data_augmentation(flip, input_shape, extra_layers)`: Returns a keras.Sequential object containing basic data augmentation layers.

## Data Utils
Functions:
- `save_model(model, model_name)`: Saves a model to the given directory.
- `load_model(model_name)`: Loads a model from the given directory.
- `log_results(history, hps, model_name)`: Logs the results of a model to a csv file with the hyperparameters.
- `ensure_directory_exists(path)`: Ensures that a directory exists at the given path. If it doesn't exist, it creates it.
- `get_tensorboard_callback(model_name)`: Returns a TensorBoard callback for a model with the given name.

## COCO Utils
Functions:
- `process_data_from_class_subset(label_subset, dataset_title)`: Processes the COCO dataset to only include images that are labeled with one of the labels in the label subset. The images are cropped to only include the object they are labeled with.
