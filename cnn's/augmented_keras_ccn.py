from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import numpy as np
import pickle

batch_size = 32
num_classes = 15
epochs = 10
img_x, img_y = 224, 224

with open('Documents/Plantr/data_arrays/my_augmented_image_dataset', 'rb') as data:
    all_final_images = pickle.load(data)

with open('Documents/Plantr/data_arrays/my_augmented_label_dataset', 'rb') as data:
    all_image_labels = pickle.load(data)

class_names = ("aloe_vera", "arrowhead_plant", "boston_fern",
               "caladium_plant", "croton_plant", "elephants_ear_plant",
               "jade_plant", "lucky_bamboo", "money_tree",
               "parlor_palm", "peace_lily", "sago_palm",
               "snake_plant", "spider_plant", "venus_fly_trap")

class_care = ("bright indirect sunlight, water deeply but infrequently, keep in temperatures(F) between 55 and 80", 
              "bright indirect sunlight, let soil dry out slightly between waterings and water less during winter", 
              "keep in cool temperatures with high humidity and indirect sunlight, keep soil damp constantly",
               "medium light with protection from midday sun, keep humidity very high, water when soil is dry to touch", 
               "water thoroughly only when soil is dry to touch, keep in temperatures(F) above 60", 
               "requires high humidity, only give indirect sun, water regularly",
               "needs 4 hours of direct sunlight each day, keep soil moist but not wet, avoid water on the leaves and only water with filtered water", 
               "water sparingly, keep out of direct sunlight, remove dead leaves", 
               "give bright and indirect light, keep in moderate humidity, let soil dry out between waterings but water thoroughly",
               "keep in high humidity and low light, water freely but reduce in the winter, keep soil moist but well drained", 
               "keep in indirectly well-lit area, keep soil moist, give high humidity and mist their leaves, keep in temperatures(F) above 60", 
               "keep in warm and humid conditions, let it dry out between waterings and don't water often",
               "keep in indirect light, allow soil to dry between waterings, avoid getting leaves wet", 
               "keep in indirect light, water well but don't allow sogginess, maintain average room temperature and humidity ", 
               "give 12 hours of direct sunlight, feed insects if they appear unhealthy, water with distilled or rainwater")

class_labels = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

train_images = np.asarray(all_final_images[:530]+all_final_images[663:1194] +
                          all_final_images[1327:1858])
train_labels = (all_image_labels[:530]+all_image_labels[663:1194] +
                all_image_labels[1327:1858])
test_images = np.asarray(all_final_images[530:663]+all_final_images[1194:1327] +
                         all_final_images[1858:])
test_labels = (all_image_labels[530:663]+all_image_labels[1194:1327] +
               all_image_labels[1858:])

train_images = train_images.reshape(train_images.shape[0], img_x, img_y, 1)
test_images = test_images.reshape(test_images.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
print('train_images shape:', train_images.shape)
print(train_images.shape[0], 'train samples')
print(test_images.shape[0], 'test samples')

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_images, test_labels),
          callbacks=[history])
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

model.save('Documents/Plantr/models/augmented_keras_cnn_model.h5')
