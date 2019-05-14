from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import user_image_writer
import statistics
from statistics import mode

img_x, img_y = 224, 224

model = tf.keras.models.load_model(
    'Documents/Plantr/models/augmented_keras_cnn_model.h5')

user_image_writer

with open('Documents/Plantr/data_arrays/user_image_data', 'rb') as data:
    user_images = pickle.load(data)

user_images = np.asarray(user_images)

user_images = user_images.reshape(user_images.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)
user_images = user_images.astype('float32')

print(' user_images shape:',  user_images.shape)
print(user_images.shape[0], 'user samples')

predictions = model.predict(user_images, verbose=1, batch_size=1)

class_names = ("aloe vera", "arrowhead plant", "boston fern",
               "caladium plant", "croton plant", "elephants ear plant",
               "jade plant", "lucky bamboo", "money tree",
               "parlor palm", "peace lily", "sago palm",
               "snake plant", "spider plant", "venus fly trap")

print(class_names[int(mode(predictions))])
