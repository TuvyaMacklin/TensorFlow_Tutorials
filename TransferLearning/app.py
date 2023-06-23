import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from utils import model_utils as mutils
from utils import plot_utils as putils
from utils import data_utils as dutils

MODEL_NAME = "TransferLearningTutorial-fine_tuned"

# Load the testing data from the validation data
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

val_ds = mutils.get_split_of_dataset_from_directory(PATH, "validation", **kwargs)


# Divide the validation set into a validation and test set.
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 5)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

test_ds = test_ds.prefetch(buffer_size = AUTOTUNE)

# Load the model
model = dutils.load_model(MODEL_NAME)

# Get a batch of images and labels. Make predictions on the images
image_batch, label_batch = test_ds.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply sigmoid to convert from logits to numbers.
# Normalize to either 0 or 1
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

# Print the predictions and the labels
print("Predictions:\n", predictions.numpy())
print("Labels:\n", label_batch)

# Display the predictions
class_names = ["Cat", "Dog"]


plt.figure(figsize = (7, 7))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    title = "Label: " + class_names[label_batch[i]] + "\n"
    title += "Prediction: " + class_names[predictions.numpy()[i]]
    plt.title(title)
    plt.axis("off")
    plt.subplots_adjust(hspace = .5)

plt.savefig("plots/predictions.png")