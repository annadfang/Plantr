from __future__ import absolute_import, division, print_function

import pathlib
import pickle
import random
import matplotlib.pyplot as plt
import IPython.display as display

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def rgb_gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


data_root = pathlib.Path("/home/jamie/Documents/Plantr/user_input_images")

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

tf.InteractiveSession()
all_final_images = []
for n in range(len(all_image_paths)):
    img_tensor = load_and_preprocess_image(all_image_paths[n])
    img_array = img_tensor.eval()
    gray_img = rgb_gray(img_array)
    all_final_images.append(gray_img)

with open('Documents/Plantr/data_arrays/user_image_data', 'wb') as output:
    pickle.dump(all_final_images, output)
