import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers

import plot_utils

import os

data_dir = "/home/ec2-user/Documents/datasets/coco"
model_dir = "/home/ec2-user/Documents/Repos/TensorFlow_Tutorials/models"
model_name = "coco_model"

# Define Hyperparameters
hp = {
    "img_height": 128,
    "img_width": 128,
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 0.0005
}

def get_split_of_dataset_from_directory(data_dir, split, **kwargs):
    ds = keras.utils.image_dataset_from_directory(
        data_dir + "/" + split,
        **kwargs
    )
    return ds

kwargs = {
    "seed": 123,
    "image_size": (hp["img_height"], hp["img_width"]),
    "batch_size": hp["batch_size"]
}

train_ds = get_split_of_dataset_from_directory(data_dir, "train", **kwargs)
val_ds = get_split_of_dataset_from_directory(data_dir, "validation", **kwargs)
class_names = train_ds.class_names
num_classes = len(class_names)

# Build the model
model = keras.Sequential()

data_augmentation_layers = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape = (hp["img_height"], hp["img_width"], 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

convolution_layers = keras.Sequential([
    layers.Conv2D(16, 3, padding= "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding= "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding= "same", activation= "relu"),
    layers.MaxPooling2D()
])

result_layers = keras.Sequential([
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation= "relu"),
    layers.Dense(num_classes)
])

model.add(data_augmentation_layers)
model.add(layers.Rescaling(1./255, input_shape = (hp["img_height"], hp["img_width"], 3)))
model.add(convolution_layers)
model.add(result_layers)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])

# Train the model
history = model.fit(
    train_ds,
    validation_data = val_ds,
    shuffle = True,
    epochs = hp["epochs"]
)

# Display the results
plot_utils.plot_history(history, to_file=True)

# Save the model
#model.save(os.path.join(model_dir, model_name, "model.h5"))

# Log model results to a csv file with the hyperparameters
def log_results(history, hps, model_name, model_dir = "/home/ec2-user/Documents/Repos/TensorFlow_Tutorials/models"):
    # Get the final training and validation accuracy
    accuracy = history.history["accuracy"][-1]
    val_accuracy = history.history["val_accuracy"][-1]

    # Ensure that a directory exists for the model results
    # The path should be model_dir/model_name/logs.csv
    ensure_directory_exists(os.path.join(model_dir, model_name))
    csv_path = os.path.join(model_dir, model_name, "logs.csv")

    # Create a csv file if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("date and time,accuracy,val_accuracy,img_height,img_width,batch_size,epochs,learning_rate\n")

    # Append the results to the csv file
    with open(csv_path, "a") as f:
        f.write(f"{datetime.datetime.now()},{accuracy},{val_accuracy},{hps['img_height']},{hps['img_width']},{hps['batch_size']},{hps['epochs']},{hps['learning_rate']}\n")

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

# Log the results
log_results(history, hp, model_name)

