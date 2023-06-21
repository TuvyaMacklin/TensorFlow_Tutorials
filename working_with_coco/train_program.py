import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers

from utils import plot_utils as putils
from utils import model_data_utils as mutils

data_dir = "/home/ec2-user/Documents/datasets/coco"
model_name = "coco_single_classification"

# Define Hyperparameters
hp = {
    "img_height": 128,
    "img_width": 128,
    "batch_size": 32,
    "epochs": 30,
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

# Build the model
model = keras.Sequential()

data_augmentation_layers = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape = (hp["img_height"], hp["img_width"], 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

convolution_layers_1 = keras.Sequential([
    layers.Conv2D(32, 3, padding= "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding= "same", activation= "relu"),
    layers.MaxPooling2D(),
])

convolution_layers_2 = keras.Sequential([
    layers.Dropout(0.1),
    layers.Conv2D(32, 3, padding= "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding= "same", activation= "relu"),
    layers.MaxPooling2D(),
])

result_layers = keras.Sequential([
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(128, activation= "relu"),
    layers.Dense(num_classes)
])

model.add(data_augmentation_layers)
model.add(layers.Rescaling(1./255))
model.add(convolution_layers_1)
model.add(convolution_layers_2)
model.add(result_layers)

model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])

# Set up TensorBoard
tensorboard_callback = mutils.get_tensorboard_callback(model_name)

# Train the model
history = model.fit(
    train_ds,
    validation_data = val_ds,
    shuffle = True,
    epochs = hp["epochs"],
    callbacks = [tensorboard_callback]
)

model.evaluate(test_ds)

# Display the results
putils.plot_history(history, to_file=True)

# Save the model and log the results
mutils.save_model(model, model_name)
mutils.log_results(history, hp, model_name)