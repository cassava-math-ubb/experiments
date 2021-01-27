from skimage import io
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

base_path = '/Volumes/LaCie SSD/cassava-leaf-disease-classification/'
train_path = base_path + 'train_images/'

img_generator = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True,
                           shear_range=0.2, zoom_range=0.1, fill_mode='reflect')

filename = train_path + os.listdir(train_path)[0]
image = io.imread(filename)
input_shape = image.shape
# plt.imshow(img_generator.random_transform(image))
# plt.show()

# ----

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# ----

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
batch_size = 16

# ----

train_path = base_path + 'flow/'
train_image_gen = img_generator.flow_from_directory(train_path, target_size=input_shape[:2], color_mode='rgb',
                                                    batch_size=batch_size, class_mode='sparse')

test_path = base_path + 'test flow/'
test_image_gen = img_generator.flow_from_directory(test_path, target_size=input_shape[:2], color_mode='rgb',
                                                    batch_size=batch_size, class_mode='sparse')

print(train_image_gen.class_indices)

results = model.fit_generator(train_image_gen, epochs=10, validation_data=test_image_gen, callbacks=[early_stopping])

# to do: model evaluation after Statistics exam :D