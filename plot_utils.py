'''
Utility functions to plot the history of a tensorflow model

Author: Tuvya Macklin

Date: 2023-06-13

Functions:
    plot_accuracy(history) - Creates a plot of the accuracy of a model with respect to the epochs.
'''
import matplotlib.pyplot as plt

def plot_accuracy(history):
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
    plt.show()

def plot_history(history, aspects = ["accuracy"], height = 7, length = 7, to_file = False):
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
    
    plt.show()


def plot_images(rows, cols, images, labels = None, height = 7, length = 7, to_file = False):
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
        plt.savefig("images.png")
    else:
        plt.show()