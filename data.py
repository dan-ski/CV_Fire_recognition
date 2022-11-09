import cv2
import glob
import pandas as pd
import numpy as np
from tensorflow import keras

PATH_FIRE = r'C:\Users\szturminskid\Documents\Order Soft\Daniel\Fire\archive\fire_dataset\fire_images'

PATH_NOFIRE = r'C:\Users\szturminskid\Documents\Order Soft\Daniel\Fire\archive\fire_dataset\non_fire_images'

fire_image_example = r'C:\Users\szturminskid\Documents\Order Soft\Daniel\Fire\archive\fire_dataset\fire_images\fire.1.png'

# https://stackoverflow.com/questions/33369832/read-multiple-images-on-a-folder-in-opencv-python

# images = [cv2.imread(file) for file in glob.glob("path/to/files/*.png")]

def load_data(path):
    fire_img = cv2.imread(fire_image_example)

    return fire_img

fire_image = load_data(fire_image_example)

window_name = 'image'

cv2.imshow(window_name, fire_image)
cv2.waitKey(0)
cv2.destroyAllWindows()