from utils.data_utils import load_model
import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt
import os

model = load_model("unet_model")

# Load and normalize the images
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image(datapoint):
    input_image = tf.image.resize(datapoint["image"], (128, 128))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (128, 128),
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

# Display an example with its mask
def display(display_list, to_file = True, root_dir = "plots", file_name = "img_and_mask", count = None):
    '''
    Display a list of images and their masks

    Args:
        display_list (list): A list of images and their masks. The image and mask should be numpy arrays.
        to_file (bool): Whether to save the image to a file or not
        root_dir (str): The root directory to save the image to
        file_name (str): The name of the file to save the image to
        count (int): The number of the image to save
    
    Returns:
        None

    Side Effects:
        Saves the image to a file
    '''

    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    
    if to_file:
        path = os.path.join(root_dir, file_name)
        if not count is None:
            path += "-" + str(count)

        path += ".png"
        plt.savefig(path)
    else:
        plt.show()

# Evaluate the model
dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info = True)
test_ds = dataset["test"].map(load_image, num_parallel_calls = tf.data.AUTOTUNE)

one_example = test_ds.take(1)

for i, (image, mask) in enumerate(one_example):
    sample_image, sample_mask = image, mask
    display([sample_image, sample_mask], count = i)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis = -1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset = test_ds, num = 1):
    for index, (image, mask) in enumerate(dataset.take(num)):
        pred_mask = model.predict(image[tf.newaxis, ...])
        display([image, mask, create_mask(pred_mask)], count=index)

show_predictions(num = 3)