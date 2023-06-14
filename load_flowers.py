import PIL
import pathlib
import tensorflow as tf

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin = dataset_url, untar = True)
data_dir = pathlib.Path(data_dir)

roses = list(data_dir.glob("roses/*"))
PIL.Image.open(str(roses[0])).show()

print("showed a rose")