'''
Utility functions to plot the history of a tensorflow model

Author: Tuvya Macklin

Date: 06-13-2023

Functions:
    `plot_accuracy(history)` - Creates a plot of the accuracy of a model with respect to the epochs.
    `plot_history(history)` - Creates a plot of the history of a model with respect to the epochs.
    `plot_images(rows, cols, images, label)` - Plots a grid of images.
    `plot_examples_from_dataset(dataset, num_of_examples)` - Plots examples from a dataset.
'''
from math import ceil, sqrt
import matplotlib.pyplot as plt

import os

root_dir = "/home/ec2-user/Documents/Repos/TensorFlow_Tutorials/plots"

def plot_accuracy(history, to_file = False, file_name = "accuracy.png"):
    '''
    Creates a plot of the accuracy of a model with respect to the epochs.
    Plots the test accuracy and the validation accuracy.

    Parameters:
        history - The history of the model.
    '''
    plt.plot(history.history["accuracy"], label = "accuracy")
    plt.plot(history.history["val_accuracy"], label = "val_accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend(loc = "lower right")
    
    if to_file:
        plt.savefig(os.path.join(root_dir, file_name))
    else:
        plt.show()

def plot_history(history, aspects = ["accuracy"], height = 7, length = 7, to_file = False, file_name = "history.png"):
    plt.figure(figsize = (height, length))

    for count, aspect in enumerate(aspects):
        plt.subplot(1, len(aspects), count + 1)
        plt.plot(history.history[aspect], label = "Training " + aspect)
        plt.plot(history.history["val_" + aspect], label = "Validation " + aspect)

        plt.xlabel("Epoch")
        plt.ylabel(aspect)

        plt.title("Training and Validation " + aspect)

        if aspect == "loss":
            plt.legend(loc = "upper right")
        else:
            plt.legend(loc = "lower right")
    
    if to_file:
        plt.savefig(os.path.join(root_dir, file_name))
    else:
        plt.show()


def plot_images(rows, cols, images, labels = None, height = 7, length = 7, to_file = False, file_name = "images.png"):
    '''
    Plots a grid of images.

    Parameters:
        rows - The number of rows in the grid.
        cols - The number of columns in the grid.
        images - The images to plot.
        labels - The labels of the images.
        height - The height of the plot.
        length - The length of the plot.
        to_file - Whether or not to save the plot to a file.

    Returns:
        None
    
    Example:
        plot_images(3, 3, images, labels, 10, 10, True)

        This will plot a 3x3 grid of images with the labels below them.
        The plot will be 10 inches by 10 inches.
        The plot will be saved to a file.
    '''

    plt.figure(figsize = (height, length))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)

        plt.imshow(images[i])
        if not labels is None:
            plt.title(labels[i])
        plt.axis("off")
    
    if to_file:
        plt.savefig(os.path.join(root_dir, file_name))
    else:
        plt.show()

def plot_examples_from_dataset(dataset, num_of_examples, to_file = False, file_name = "examples.png"):
    '''
    Plots examples from a dataset.

    Parameters:
        dataset - The dataset to plot examples from.
        num_of_examples - The number of examples to plot.
    
    Returns:
        None

    Side Effects:
        Plots the examples from the dataset.
    '''

    cols = int(sqrt(num_of_examples))
    rows = ceil(num_of_examples / cols)

    class_names = dataset.class_names

    plt.figure(figsize = (10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_of_examples):
            if i >= len(images):
                break
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    
    if to_file:
        plt.savefig(os.path.join(root_dir, file_name))
    else:
        plt.show()