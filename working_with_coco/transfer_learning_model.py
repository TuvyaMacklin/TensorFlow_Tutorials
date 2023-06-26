import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers

from utils import plot_utils as putils
from utils import data_utils as dutils
from utils import model_utils as mutils

data_dir = "/home/ec2-user/Documents/datasets/coco_sports"
model_name = "coco_sports_transfer_learning"

# Define Hyperparameters
hp = {
    "img_height": 128,
    "img_width": 128,
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001
}

def get_split_of_dataset_from_directory(data_dir, split, **kwargs):
    ds = keras.utils.image_dataset_from_directory(
        data_dir + "/" + split,
        **kwargs
    )
    return ds

train_args = {
    "seed": 123,
    "image_size": (hp["img_height"], hp["img_width"]),
    "batch_size": hp["batch_size"],
    "shuffle": True,
    "validation_split": 0.2,
    "subset": "training"
}

val_args = {
    "seed": 123,
    "image_size": (hp["img_height"], hp["img_width"]),
    "batch_size": hp["batch_size"],
    "shuffle": True,
    "validation_split": 0.2,
    "subset": "validation"
}

test_args = {
    "seed": 123,
    "image_size": (hp["img_height"], hp["img_width"]),
    "batch_size": hp["batch_size"],
    "shuffle": True
}

train_ds = get_split_of_dataset_from_directory(data_dir, "train", **train_args)
val_ds = get_split_of_dataset_from_directory(data_dir, "train", **val_args)
test_ds = get_split_of_dataset_from_directory(data_dir, "validation", **test_args)
class_names = train_ds.class_names
num_classes = len(class_names)

augmentation_layers = mutils.get_basic_data_augmentation()

# Preprocess data to fit into the MobileNetV2 model
preprocessing_layers = tf.keras.applications.mobilenet_v2.preprocess_input

# Load the pre-trained MobileNet V2 model
# From this the output shape is (32, 5, 5, 1280)
IMG_SHAPE = (128, 128, 3)
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
prediction_layer = tf.keras.layers.Dense(10)

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
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])

# Set up TensorBoard
tensorboard_callback = dutils.get_tensorboard_callback(model_name)

model.summary()
quit()

# Train the model
history = model.fit(
    train_ds,
    validation_data = val_ds,
    shuffle = True,
    epochs = hp["epochs"],
    callbacks = [tensorboard_callback]
)

model.evaluate(test_ds)

# Fine tune the model
base_model.trainable = True

fine_tune_limit = 100

for layer in base_model.layers[:fine_tune_limit]:
    layer.trainable = False

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = hp["learning_rate"] / 10),
    metrics = ["accuracy"]
)

fine_tune_epochs = 10
total_epochs = hp["epochs"] + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs = total_epochs,
    initial_epoch = history.epoch[-1],
    callbacks = [tensorboard_callback],
    validation_data = val_ds
)

# Save the model and log the results
dutils.save_model(model, model_name + "_fine_tuned")
model.evaluate(test_ds)