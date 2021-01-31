from keras import models
from datetime import datetime
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from confidence_interval import confidence_interval

WORK_DIR = '/home/teofana/cassava leafs'
SAVE_DIR = '/saved_models/'
TARGET_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 2


def create_model():
    _model = models.Sequential()

    _model.add(EfficientNetB0(include_top=False, weights='imagenet',
                              input_shape=(TARGET_SIZE, TARGET_SIZE, 3)))

    _model.add(layers.GlobalAveragePooling2D())
    _model.add(layers.Dense(5, activation="softmax"))

    _model.compile(optimizer=Adam(lr=0.001),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return _model


def plot_history(_history):
    # summarize history for accuracy
    plt.plot(_history.history['accuracy'])
    plt.plot(_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.close()

    # summarize history for loss
    plt.plot(_history.history['loss'])
    plt.plot(_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('loss.png')


if __name__ == '__main__':
    train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
    train_labels.label = train_labels.label.astype('str')

    Y = train_labels[['label']]
    n = len(train_labels)
    skf = StratifiedKFold(n_splits=5)

    valid_accuracy = []
    valid_loss = []
    fold_var = 1

    train_idg = ImageDataGenerator(
        preprocessing_function=None,
        rotation_range=45,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        shear_range=0.1,
        height_shift_range=0.1,
        width_shift_range=0.1)

    validation_idg = ImageDataGenerator(validation_split=0.2)

    for train_index, val_index in skf.split(np.zeros(n), Y):
        training_data = train_labels.iloc[train_index]
        validation_data = train_labels.iloc[val_index]

        train_generator = train_idg.flow_from_dataframe(training_data,
                                                        directory=os.path.join(WORK_DIR, "train_images"),
                                                        subset="training",
                                                        x_col="image_id",
                                                        y_col="label",
                                                        target_size=(TARGET_SIZE, TARGET_SIZE),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode="sparse")

        validation_generator = validation_idg.flow_from_dataframe(validation_data,
                                                                  directory=os.path.join(WORK_DIR, "train_images"),
                                                                  subset="validation",
                                                                  x_col="image_id",
                                                                  y_col="label",
                                                                  target_size=(TARGET_SIZE, TARGET_SIZE),
                                                                  batch_size=BATCH_SIZE,
                                                                  class_mode="sparse")

        model = create_model()
        model.summary()

        model_save = ModelCheckpoint(f"saved_models/best_model_{str(fold_var)}.h5",
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss',
                                     mode='min')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   patience=5,
                                   mode='min',
                                   restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.3,
                                      patience=2,
                                      min_delta=0.001,
                                      mode='min')
        tensorboard = TensorBoard(log_dir=f"logs/scalars/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[model_save, early_stop, reduce_lr, tensorboard],
            verbose=0
        )

        # save model
        model_json = model.to_json()
        with open(f"saved_models/model_{str(fold_var)}.json", "w") as json_file:
            json_file.write(model_json)

        # evaluate validation dataset using the best weights
        model.load_weights(f"saved_models/best_model_{str(fold_var)}.h5")
        results = model.evaluate(validation_generator)
        results = dict(zip(model.metrics_names, results))

        valid_accuracy.append(results['accuracy'])
        valid_loss.append(results['loss'])

        tf.keras.backend.clear_session()

        fold_var += 1

    print(f"CI loss: {confidence_interval(valid_loss)}")
    print(f"CI acc: {confidence_interval(valid_accuracy)}")
