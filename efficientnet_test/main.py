from keras import models
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

WORK_DIR = '/home/teofana/cassava leafs'

train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_labels.head()
train_labels.label = train_labels.label.astype('str')

BATCH_SIZE = 2
STEPS_PER_EPOCH = len(train_labels) * 0.8 / BATCH_SIZE
VALIDATION_STEPS = len(train_labels) * 0.2 / BATCH_SIZE
EPOCHS = 10
TARGET_SIZE = 512

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   preprocessing_function=None,
                                   rotation_range=45,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest',
                                   shear_range=0.1,
                                   height_shift_range=0.1,
                                   width_shift_range=0.1)

train_generator = train_datagen.flow_from_dataframe(train_labels,
                                                    directory=os.path.join(WORK_DIR, "train_images"),
                                                    subset="training",
                                                    x_col="image_id",
                                                    y_col="label",
                                                    target_size=(TARGET_SIZE, TARGET_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="sparse")

validation_datagen = ImageDataGenerator(validation_split=0.2)

validation_generator = validation_datagen.flow_from_dataframe(train_labels,
                                                              directory=os.path.join(WORK_DIR, "train_images"),
                                                              subset="validation",
                                                              x_col="image_id",
                                                              y_col="label",
                                                              target_size=(TARGET_SIZE, TARGET_SIZE),
                                                              batch_size=BATCH_SIZE,
                                                              class_mode="sparse")


def create_model():
    model = models.Sequential()

    model.add(EfficientNetB0(include_top=False, weights='imagenet',
                             input_shape=(TARGET_SIZE, TARGET_SIZE, 3)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(5, activation="softmax"))

    model.compile(optimizer=Adam(lr=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


model = create_model()
model.summary()

model_save = ModelCheckpoint('./best_model.h5',
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001,
                           patience=5, mode='min', verbose=1,
                           restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=2, min_delta=0.001,
                              mode='min')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr]
)

# evaluate model
score = model.evaluate_generator(validation_generator)
print('Score:', score)

# save model
model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)
model.save_weights('weights.h5')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy.png')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.png')
