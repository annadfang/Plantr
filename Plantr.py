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

class_names = ("aloe vera", "arrowhead plant", "boston fern",
               "caladium plant", "croton plant", "elephants ear plant",
               "jade plant", "lucky bamboo", "money tree",
               "parlor palm", "peace lily", "sago palm",
               "snake plant", "spider plant", "venus fly trap")

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

model = tf.keras.models.load_model(
    'Documents/Plantr/models/augmented_keras_cnn_model.h5')

user_image_writer

with open('Documents/Plantr/data_arrays/user_image_data', 'rb') as data:
    user_images = pickle.load(data)

user_images_array = np.asarray(user_images)

user_images = user_images_array.reshape(
    user_images_array.shape[0], img_x, img_y, 1)
user_images = user_images.astype('float32')

predictionarrays = model.predict(user_images, verbose=0, batch_size=1)

predictions = []
for i in predictionarrays:
    predictions.append(np.argmax(i))

print('''
          _____                    _____            _____                    _____                _____          
         /\    \                  /\    \          /\    \                  /\    \              /\    \         
        /::\    \                /::\____\        /::\    \                /::\____\            /::\    \        
       /::::\    \              /:::/    /       /::::\    \              /::::|   |            \:::\    \       
      /::::::\    \            /:::/    /       /::::::\    \            /:::::|   |             \:::\    \      
     /:::/\:::\    \          /:::/    /       /:::/\:::\    \          /::::::|   |              \:::\    \     
    /:::/__\:::\    \        /:::/    /       /:::/__\:::\    \        /:::/|::|   |               \:::\    \    
   /::::\   \:::\    \      /:::/    /       /::::\   \:::\    \      /:::/ |::|   |               /::::\    \   
  /::::::\   \:::\    \    /:::/    /       /::::::\   \:::\    \    /:::/  |::|   | _____        /::::::\    \  
 /:::/\:::\   \:::\____\  /:::/    /       /:::/\:::\   \:::\    \  /:::/   |::|   |/\    \      /:::/\:::\    \ 
/:::/  \:::\   \:::|    |/:::/____/       /:::/  \:::\   \:::\____\/:: /    |::|   /::\____\    /:::/  \:::\____\\
\::/    \:::\  /:::|____|\:::\    \       \::/    \:::\  /:::/    /\::/    /|::|  /:::/    /   /:::/    \::/    /
 \/_____/\:::\/:::/    /  \:::\    \       \/____/ \:::\/:::/    /  \/____/ |::| /:::/    /   /:::/    / \/____/ 
          \::::::/    /    \:::\    \               \::::::/    /           |::|/:::/    /   /:::/    /          
           \::::/    /      \:::\    \               \::::/    /            |::::::/    /   /:::/    /           
            \::/____/        \:::\    \              /:::/    /             |:::::/    /    \::/    /            
             ~~               \:::\    \            /:::/    /              |::::/    /      \/____/             
                               \:::\    \          /:::/    /               /:::/    /                           
                                \:::\____\        /:::/    /               /:::/    /                            
                                 \::/    /        \::/    /                \::/    /                             
                                  \/____/          \/____/                  \/____/                              
''')
# print('\n')
print('\n')
print('                                         ' +
      class_names[mode(predictions)])
# print('\n')
# print('\n')

print('''
          _____                    _____                    _____                    _____          
         /\    \                  /\    \                  /\    \                  /\    \         
        /::\    \                /::\    \                /::\    \                /::\    \        
       /::::\    \              /::::\    \              /::::\    \              /::::\    \       
      /::::::\    \            /::::::\    \            /::::::\    \            /::::::\    \      
     /:::/\:::\    \          /:::/\:::\    \          /:::/\:::\    \          /:::/\:::\    \     
    /:::/  \:::\    \        /:::/__\:::\    \        /:::/__\:::\    \        /:::/__\:::\    \    
   /:::/    \:::\    \      /::::\   \:::\    \      /::::\   \:::\    \      /::::\   \:::\    \   
  /:::/    / \:::\    \    /::::::\   \:::\    \    /::::::\   \:::\    \    /::::::\   \:::\    \  
 /:::/    /   \:::\    \  /:::/\:::\   \:::\    \  /:::/\:::\   \:::\____\  /:::/\:::\   \:::\    \ 
/:::/____/     \:::\____\/:::/  \:::\   \:::\____\/:::/  \:::\   \:::|    |/:::/__\:::\   \:::\____\\
\:::\    \      \::/    /\::/    \:::\  /:::/    /\::/   |::::\  /:::|____|\:::\   \:::\   \::/    /
 \:::\    \      \/____/  \/____/ \:::\/:::/    /  \/____|:::::\/:::/    /  \:::\   \:::\   \/____/ 
  \:::\    \                       \::::::/    /         |:::::::::/    /    \:::\   \:::\    \     
   \:::\    \                       \::::/    /          |::|\::::/    /      \:::\   \:::\____\    
    \:::\    \                      /:::/    /           |::| \::/____/        \:::\   \::/    /    
     \:::\    \                    /:::/    /            |::|  ~|               \:::\   \/____/     
      \:::\    \                  /:::/    /             |::|   |                \:::\    \         
       \:::\____\                /:::/    /              \::|   |                 \:::\____\        
        \::/    /                \::/    /                \:|   |                  \::/    /        
         \/____/                  \/____/                  \|___|                   \/____/         
''')
# print('\n')
print('\n')
print(class_care[mode(predictions)])
# print('\n')
# print('\n')


def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label], 100*np.max(predictions_array), class_names[predicted_label]))


def plot_value_array(i, predictions_array):
    predictions_array = predictions_array[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(15), predictions_array, color="blue")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')


num_rows = int(np.sqrt(user_images_array.shape[0]))
num_cols = user_images_array.shape[0]-int(np.sqrt(user_images_array.shape[0]))
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictionarrays, user_images_array)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictionarrays)
plt.show()
