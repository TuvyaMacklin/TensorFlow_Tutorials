import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

import pathlib

import plot_utils

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin = dataset_url, untar = True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

class_names = train_ds.class_names

data = [(images, labels) for images, labels in train_ds.take(1)]

images = [image.numpy().astype("uint8") for image in data[0][0]]
labels = [class_names[label] for label in data[0][1]]

# Augment the data

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape = (img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

data_augmentation.summary()

augmented_images = []
for i in range(9):
    augmented_image_ds = data_augmentation(data[0][0])
    augmented_image = augmented_image_ds[0].numpy().astype("uint8")
    augmented_images.append(augmented_image)

# plot_utils.plot_images(3, 3, augmented_images)

# Configure the data set to be stored in memory
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

# Build the model
num_classes = len(class_names)
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape = (img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding = "same", activation = "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = "same", activation = "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = "same", activation = "relu"),
    layers.MaxPooling2D(),

    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation = "relu"),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer = "adam",
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])

model.summary()

# Train the model
epochs = 30
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
)

plot_utils.plot_history(history, aspects = ["accuracy","loss"], to_file = True)