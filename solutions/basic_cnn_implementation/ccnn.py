from skimage import io
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_path = '/Volumes/LaCie SSD/cassava-leaf-disease-classification/'
train_path = base_path + 'train_images/'

img_generator = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True,
                           shear_range=0.2, zoom_range=0.1, fill_mode='reflect')

filename = train_path + os.listdir(train_path)[0]
print(filename)
image = io.imread(filename)
plt.imshow(img_generator.random_transform(image))
plt.show()