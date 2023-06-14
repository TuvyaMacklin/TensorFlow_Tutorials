import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import plot_utils

print(tf.__version__)

# import and normalize the data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

print(train_labels[0])

# Display 25 images
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(10,10))

# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", input_shape = (32, 32, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Add the convolution layers
model = models.Sequential()
model.add(data_augmentation)
model.add(layers.Conv2D(32, (3,3), activation = "relu", input_shape = (32, 32, 3)))
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))

# Add dropout layer
model.add(layers.Dropout(0.2))

# Add the dense layers to get an output
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"])

history = model.fit(train_images, train_labels, epochs = 100, validation_split = 0.1)

plot_utils.plot_history(history, aspects = ["accuracy", "loss"], to_file = True)