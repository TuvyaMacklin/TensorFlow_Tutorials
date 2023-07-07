import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt

import utils.data_utils as dutils

# Load the dataset
dataset, info = tfds.load("oxford_iiit_pet:3.*.*", with_info = True)

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

TRAIN_LENGTH = info.splits["train"].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_images = dataset["train"].map(load_image, num_parallel_calls = tf.data.AUTOTUNE)
test_images = dataset["test"].map(load_image, num_parallel_calls = tf.data.AUTOTUNE)

# Augment the images

class AugmentLayer(tf.keras.layers.Layer):
    def __init__(self, seed = 42):
        super().__init__()

        # Both layers should have the same seed so they augment in tandem
        self.augment_inputs = tf.keras.layers.RandomFlip(mode = "horizontal", seed = seed)
        self.augment_mask = tf.keras.layers.RandomFlip(mode = "horizontal", seed = seed)

    def call(self, inputs, mask):
        inputs = self.augment_inputs(inputs)
        mask = self.augment_mask(mask)

        return inputs, mask
    
train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(AugmentLayer())
    .prefetch(buffer_size = tf.data.AUTOTUNE)
)

test_batches = test_images.batch(BATCH_SIZE)

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

for i, (images, masks) in enumerate(train_batches.take(2)):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask], count = i)

# Define the model
base_model = tf.keras.applications.MobileNetV2(input_shape = [128, 128, 3], include_top = False)

layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project',
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs = base_model.input, outputs = base_model_outputs)
down_stack.trainable = False

# Define the upstack
upstack = [
    pix2pix.upsample(512, 3),
    pix2pix.upsample(256, 3),
    pix2pix.upsample(128, 3),
    pix2pix.upsample(64, 3),
]

# Put it all together
def get_unet_model(output_channels: int):
    inputs = tf.keras.layers.Input(shape = [128, 128, 3])

    # Downsampling
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling connecting in the skips
    for up, skip in zip(upstack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    
    # Define output layer
    last = tf.keras.layers.Conv2DTranspose(
        filters = output_channels,
        kernel_size = 3,
        strides = 2,
        padding = "same"
    )

    x = last(x)

    return tf.keras.Model(inputs = inputs, outputs = x)

# Compile the model
OUTPUT_CLASSES = 3

model = get_unet_model(OUTPUT_CLASSES)
model.compile(
    optimizer = "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ["accuracy"]
)


# See what the model predicts before training
def create_mask(prediction_mask):
    prediction_mask = tf.math.argmax(prediction_mask, axis = -1)
    prediction_mask = prediction_mask[..., tf.newaxis]
    return prediction_mask[0]

def show_predictions(dataset = None, num = 1):
    if dataset:
        for index, (image, mask) in enumerate(dataset.take(num)):
            prediction = model.predict(image)
            display([image[0], mask[0], create_mask(prediction)], file_name = "predicted_mask", count = index)
    else:
        display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))], file_name = "predicted_mask")
        
show_predictions()

# Train the model
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

history = model.fit(
    train_batches,
    epochs = EPOCHS,
    steps_per_epoch = STEPS_PER_EPOCH,
    validation_steps= VALIDATION_STEPS,
    validation_data = test_batches
)

show_predictions(dataset = test_batches)

dutils.save_model(model, "unet_model")