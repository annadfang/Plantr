from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('Documents/Plantr/data_arrays/my_image_dataset', 'rb') as data:
    all_final_images = pickle.load(data)

with open('Documents/Plantr/data_arrays/my_label_dataset', 'rb') as data:
    all_image_labels = pickle.load(data)

class_names = ("aloe_vera", "arrowhead_plant", "boston_fern",
               "caladium_plant", "croton_plant", "elephants_ear_plant",
               "jade_plant", "lucky_bamboo", "money_tree",
               "parlor_palm", "peace_lily", "sago_palm",
               "snake_plant", "spider_plant", "venus_fly_trap")

class_labels = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

train_images = np.asarray(all_final_images[:600])
train_labels = all_image_labels[:600]
test_images = np.asarray(all_final_images[600:])
test_labels = all_image_labels[600:]

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(224, 224)),
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(15, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

model.save('Documents/Plantr/models/nparrays_nn_model.h5')

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(15), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
