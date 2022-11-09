import cv2
import numpy as np

import glob
import pandas as pd
import numpy as np
from tensorflow import keras

#PATH_FIRE = r'C:\Users\szturminskid\Documents\Order Soft\Daniel\Fire\archive\fire_dataset\fire_images'
PATH_FIRE = r'D:\Data Science\Fire recognition- computer vision\archive\fire_dataset\fire_images'

#PATH_NOFIRE = r'C:\Users\szturminskid\Documents\Order Soft\Daniel\Fire\archive\fire_dataset\non_fire_images'
PATH_NOFIRE = r'D:\Data Science\Fire recognition- computer vision\archive\fire_dataset\non_fire_images'

fire_image_example = r'C:\Users\szturminskid\Documents\Order Soft\Daniel\Fire\archive\fire_dataset\fire_images\fire.1.png'

# https://stackoverflow.com/questions/33369832/read-multiple-images-on-a-folder-in-opencv-python

# images = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]


def load_data(images_path):
    #fire_img = cv2.imread(fire_image_example)
    images = [cv2.imread(file) for file in glob.glob(f"{images_path}/*.png")]
    return images

def show_image(image):

    window_name = 'image'
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_type(data):
    print('Type: ',type(data))

def train_test_sets(x_images, y_images):

fire_images = load_data(PATH_FIRE)
nonfire_images = load_data(PATH_NOFIRE)

show_image(fire_images[0])
print_type(fire_images)
print(len(fire_images))


