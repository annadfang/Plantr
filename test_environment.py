import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('Documents/Plantr/my_image_dataset', 'rb') as data:
    all_final_images = pickle.load(data)

with open('Documents/Plantr/my_label_dataset', 'rb') as data:
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


plt.figure(figsize=(10, 10))
for i in range(4):
    if i < 2:
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    else:
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i-2], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
plt.show()
