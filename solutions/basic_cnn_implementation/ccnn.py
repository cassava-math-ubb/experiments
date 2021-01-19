from matplotlib.image import imread
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_path = '/Volumes/LaCie SSD/cassava-leaf-disease-classification/'
train_path = base_path + 'train_images/'

imgen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1)