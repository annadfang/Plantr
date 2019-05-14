import pickle
import random
import cv2
import os
import matplotlib.pyplot as plt
import csv
import numpy as np

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNEL = 1


def zoom(image):
    zoom_pix = 10*random.randint(0, 10)
    zoom_factor = 1 + (2*zoom_pix)/IMAGE_HEIGHT
    image = cv2.resize(image, None, fx=zoom_factor,
                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    top_crop = (image.shape[0] - IMAGE_HEIGHT)//2
#     bottom_crop = image.shape[0] - top_crop - IMAGE_HEIGHT
    left_crop = (image.shape[1] - IMAGE_WIDTH)//2
#     right_crop = image.shape[1] - left_crop - IMAGE_WIDTH
    image = image[top_crop: top_crop+IMAGE_HEIGHT,
                  left_crop: left_crop+IMAGE_WIDTH]
    return image


def random_brightness(image):
    rand = random.uniform(.3, 1.0)
    rand2 = random.uniform(0, 4)
    range1 = int(np.floor(random.uniform(0, 224)))
    range2 = int(np.floor(random.uniform(0, 224)))

    new_img = image

    if rand2 < 1:
        new_img[:range1, :range2] = cv2.multiply(
            image[:range1, :range2], np.array([rand]))
    elif rand2 >= 1 and rand2 < 2:
        new_img[:range1, range2:] = cv2.multiply(
            image[:range1, range2:], np.array([rand]))
    elif rand2 >= 2 and rand2 < 3:
        new_img[range1:, :range2] = cv2.multiply(
            image[range1:, :range2], np.array([rand]))
    else:
        new_img[range1:, range2:] = cv2.multiply(
            image[range1:, range2:], np.array([rand]))

    return new_img


def flip_hor(image):
    new_img = cv2.flip(image, 1)
    return new_img


def flip_ver(image):
    new_img = cv2.flip(image, 0)
    return new_img


def flip_both(image):
    new_img = cv2.flip(image, -1)
    return new_img


with open('Documents/Plantr/data_arrays/my_image_dataset', 'rb') as data:
    all_final_images = pickle.load(data)

with open('Documents/Plantr/data_arrays/my_label_dataset', 'rb') as data:
    all_image_labels = pickle.load(data)


for i in range(len(all_image_labels)):
    label = all_image_labels[i]

    all_image_labels.extend([label, label, label, label, label])
    all_final_images.append(all_final_images[i])
    all_final_images.append(all_final_images[i])
    all_final_images.append(all_final_images[i])
    all_final_images.append(all_final_images[i])
    all_final_images.append(all_final_images[i])

with open('Documents/Plantr/data_arrays/my_augmented_image_dataset', 'wb') as output:
    pickle.dump(all_final_images, output)

with open('Documents/Plantr/data_arrays/my_augmented_label_dataset', 'wb') as output:
    pickle.dump(all_image_labels, output)
