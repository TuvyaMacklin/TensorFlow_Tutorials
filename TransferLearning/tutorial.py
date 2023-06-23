import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from utils import model_utils as mutils
from utils import plot_utils as putils
from utils import data_utils as dutils

MODEL_NAME = "TransferLearningTutorial"

URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

kwargs = {
    "shuffle": True,
    "batch_size": BATCH_SIZE,
    "image_size": IMG_SIZE
}

train_ds = mutils.get_split_of_dataset_from_directory(PATH, "train", **kwargs)
val_ds = mutils.get_split_of_dataset_from_directory(PATH, "validation", **kwargs)

# Show 9 examples from the training set.
putils.plot_examples_from_dataset(train_ds, 9, to_file = True)

# Divide the validation set into a validation and test set.
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size = AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size = AUTOTUNE)

augmentation_layers = mutils.get_basic_data_augmentation()

# Preprocess data to fit into the MobileNetV2 model
preprocessing_layers = tf.keras.applications.mobilenet_v2.preprocess_input

# Load the pre-trained MobileNet V2 model
# From this the output shape is (32, 5, 5, 1280)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape = IMG_SHAPE,
    include_top = False,
    weights = "imagenet"
)

# Freeze the mobilenet layers
base_model.trainable = False

# Flatten the output from a 5x5 matrix of 1280 dimension vectors to one 1280 dimension vector
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Create a prediction/output layer
prediction_layer = tf.keras.layers.Dense(1)

# Build the model
inputs = tf.keras.Input(shape = IMG_SHAPE)
x = augmentation_layers(inputs)
x = preprocessing_layers(x)
x = base_model(x, training = False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)

# Compile the model
base_learning_rate= 0.0001
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = base_learning_rate),
    loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
    metrics = ["accuracy"]
)

# Evaluate the model before it is trained
loss0, accuracy0 = model.evaluate(test_ds)

# Train the model
initial_epochs = 10

tensorboard_callback = dutils.get_tensorboard_callback(MODEL_NAME)

history = model.fit(
    train_ds,
    epochs = initial_epochs,
    validation_data = val_ds,
    callbacks = [tensorboard_callback]
)

# Save the model
dutils.save_model(model, MODEL_NAME)

# Evaluate the model
model.evaluate(test_ds)

# Fine tune the model. It has 154 layers
FINE_MODEL_NAME = MODEL_NAME + "-fine_tuned"
base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = base_learning_rate / 10),
    metrics = ["accuracy"]
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs = total_epochs,
    initial_epoch = history.epoch[-1],
    validation_data = val_ds,
    callbacks = [tensorboard_callback]
)

dutils.save_model(model, FINE_MODEL_NAME)
model.evaluate(test_ds)